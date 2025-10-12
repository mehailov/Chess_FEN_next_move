import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import chess
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== УЛУЧШЕННЫЙ ТОКЕНИЗАТОР ====================

class AdvancedChessTokenizer:
    def __init__(self):
        self.piece_to_idx = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }

    def fen_to_advanced_tensor(self, fen):
        """Улучшенное представление FEN с дополнительными признаками"""
        parts = fen.split(' ')
        board_part = parts[0]

        # Основная доска 8x8x12
        board_tensor = np.zeros((8, 8, 12), dtype=np.float32)

        rows = board_part.split('/')
        for i, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    board_tensor[i, col_idx, self.piece_to_idx[char]] = 1
                    col_idx += 1

        # Дополнительные признаки (21 признак)
        extra_features = np.zeros(21, dtype=np.float32)

        # Чей ход (0 - белые, 1 - черные)
        extra_features[0] = 1 if parts[1] == 'b' else 0

        # Права на рокировку
        castling = parts[2] if len(parts) > 2 else 'KQkq'
        extra_features[1] = 1 if 'K' in castling else 0  # Белые короткая
        extra_features[2] = 1 if 'Q' in castling else 0  # Белые длинная
        extra_features[3] = 1 if 'k' in castling else 0  # Черные короткая
        extra_features[4] = 1 if 'q' in castling else 0  # Черные длинная

        # Битое поле (если есть)
        if len(parts) > 3 and parts[3] != '-':
            en_passant = parts[3]
            col = ord(en_passant[0]) - ord('a')
            row = int(en_passant[1]) - 1
            en_passant_idx = row * 8 + col
            extra_features[5] = 1  # Флаг наличия битого поля
            extra_features[6] = en_passant_idx / 63.0  # Нормализованная позиция

        # Счетчики (если есть в FEN)
        if len(parts) > 4:
            try:
                halfmove_clock = int(parts[4])
                fullmove_number = int(parts[5])
                extra_features[7] = min(halfmove_clock / 50.0, 1.0)  # Нормализованный счетчик полуходов
                extra_features[8] = min(fullmove_number / 200.0, 1.0)  # Нормализованный номер хода
            except:
                pass

        # Признаки безопасности королей (грубая оценка)
        try:
            board = chess.Board(fen)
            # Позиции королей
            white_king_square = board.king(chess.WHITE)
            black_king_square = board.king(chess.BLACK)
            if white_king_square is not None:
                extra_features[9] = chess.square_file(white_king_square) / 7.0
                extra_features[10] = chess.square_rank(white_king_square) / 7.0
            if black_king_square is not None:
                extra_features[11] = chess.square_file(black_king_square) / 7.0
                extra_features[12] = chess.square_rank(black_king_square) / 7.0

            # Количество атакующих фигур вокруг королей (упрощенно)
            if white_king_square:
                attackers = board.attackers(chess.BLACK, white_king_square)
                extra_features[13] = len(attackers) / 8.0
            if black_king_square:
                attackers = board.attackers(chess.WHITE, black_king_square)
                extra_features[14] = len(attackers) / 8.0

            # Материальный баланс (упрощенно)
            white_material = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                           3 * len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                           3 * len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                           5 * len(board.pieces(chess.ROOK, chess.WHITE)) + \
                           9 * len(board.pieces(chess.QUEEN, chess.WHITE))
            black_material = len(board.pieces(chess.PAWN, chess.BLACK)) + \
                           3 * len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                           3 * len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                           5 * len(board.pieces(chess.ROOK, chess.BLACK)) + \
                           9 * len(board.pieces(chess.QUEEN, chess.BLACK))
            material_balance = (white_material - black_material) / 39.0  # Нормализация
            extra_features[15] = material_balance

        except:
            pass

        return np.concatenate([board_tensor.flatten(), extra_features])

# ==================== СЛОВАРЬ ХОДОВ ====================

class MoveVocabulary:
    def __init__(self, moves):
        self.move_to_idx = {}
        self.idx_to_move = {}
        self.build_vocab(moves)

    def build_vocab(self, moves):
        """Создает словарь всех уникальных ходов"""
        unique_moves = sorted(set(moves))
        self.move_to_idx = {move: idx for idx, move in enumerate(unique_moves)}
        self.idx_to_move = {idx: move for move, idx in self.move_to_idx.items()}

    def __len__(self):
        return len(self.move_to_idx)

    def move_to_index(self, move):
        return self.move_to_idx[move]

    def index_to_move(self, idx):
        return self.idx_to_move[idx]

# ==================== УЛУЧШЕННЫЙ DATASET ====================

class AdvancedChessDataset(Dataset):
    def __init__(self, df, tokenizer, move_vocab):
        self.df = df
        self.tokenizer = tokenizer
        self.move_vocab = move_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fen = row['fen']
        move = row['move']

        # Преобразуем FEN в тензор
        board_tensor = self.tokenizer.fen_to_advanced_tensor(fen)

        # Преобразуем ход в индекс
        move_idx = self.move_vocab.move_to_index(move)

        return torch.FloatTensor(board_tensor), torch.tensor(move_idx, dtype=torch.long)

# ==================== УЛУЧШЕННАЯ МОДЕЛЬ ====================

class ImprovedChessModel(nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size=1024):
        super().__init__()

        # Более глубокая архитектура с residual connections
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Linear(hidden_size // 4, vocab_size)

        # Инициализация весов
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2 + x1)  # residual connection
        x4 = self.layer4(x3)
        return self.output(x4)

# ==================== УЛУЧШЕННЫЙ ТРЕНИНГ ====================

def advanced_train_model(model, train_loader, val_loader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_accuracies = []
    best_accuracy = 0
    patience = 7
    patience_counter = 0

    print("Начало обучения улучшенной модели...")
    print("="*60)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        # Early stopping и сохранение лучшей модели
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': avg_loss,
            }, 'best_chess_model.pth')
            print(f"  -> Новый лучший результат! Модель сохранена.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping на эпохе {epoch+1}")
                break

    return train_losses, val_accuracies, best_accuracy

# ==================== МЕТРИКИ ====================

def calculate_top_k_accuracy(model, data_loader, k=3):
    """Вычисляет top-k accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            _, topk_pred = output.topk(k, dim=1)
            correct += topk_pred.eq(target.view(-1, 1)).sum().item()
            total += target.size(0)

    return correct / total

def calculate_perplexity(model, data_loader):
    """Вычисляет перплексию"""
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    return perplexity

# ==================== АНАЛИЗ ПРЕДСКАЗАНИЙ ====================

def analyze_predictions(model, val_loader, move_vocab, tokenizer, n_samples=5):
    """Анализ примеров предсказаний"""
    model.eval()
    data_iter = iter(val_loader)

    print("\n" + "="*60)
    print("АНАЛИЗ ПРЕДСКАЗАНИЙ:")
    print("="*60)

    for i in range(n_samples):
        data, target = next(data_iter)
        with torch.no_grad():
            output = model(data)
            probabilities = torch.softmax(output, dim=1)

        top5_probs, top5_indices = torch.topk(probabilities, 5, dim=1)

        true_move = move_vocab.index_to_move(target[0].item())
        predicted_moves = [move_vocab.index_to_move(idx.item()) for idx in top5_indices[0]]
        predicted_probs = [f"{prob:.3f}" for prob in top5_probs[0].tolist()]

        print(f"\nПример {i+1}:")
        print(f"  Правильный ход: {true_move}")
        print(f"  Топ-5 предсказаний:")
        for j, (move, prob) in enumerate(zip(predicted_moves, predicted_probs)):
            marker = " ✓" if move == true_move else ""
            print(f"    {j+1}. {move:6} ({prob}){marker}")

    # Анализ распределения уверенности модели
    print(f"\n" + "="*60)
    print("СТАТИСТИКА УВЕРЕННОСТИ:")
    print("="*60)

    all_max_probs = []
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            all_max_probs.extend(max_probs.tolist())

    all_max_probs = np.array(all_max_probs)
    print(f"Средняя уверенность: {np.mean(all_max_probs):.4f}")
    print(f"Медианная уверенность: {np.median(all_max_probs):.4f}")
    print(f"Доля предсказаний с уверенностью > 0.5: {np.mean(all_max_probs > 0.5):.4f}")
    print(f"Доля предсказаний с уверенностью > 0.8: {np.mean(all_max_probs > 0.8):.4f}")

# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_training_history(train_losses, val_accuracies):
    """Визуализация процесса обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== ОСНОВНОЙ ПАЙПЛАЙН ====================

def main():
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv('fens_training_set.csv')
    print(f"Загружено {len(df)} примеров")

    # Создание словаря ходов и токенизатора
    move_vocab = MoveVocabulary(df['move'])
    tokenizer = AdvancedChessTokenizer()

    print(f"Размер словаря ходов: {len(move_vocab)}")

    # Проверка размера входного вектора
    sample_tensor = tokenizer.fen_to_advanced_tensor(df.iloc[0]['fen'])
    input_size = len(sample_tensor)
    vocab_size = len(move_vocab)

    print(f"Размер входного вектора: {input_size}")
    print(f"Размер словаря: {vocab_size}")

    # Разделение данных
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Обучающая выборка: {len(train_df)} примеров")
    print(f"Валидационная выборка: {len(val_df)} примеров")

    # Создание datasets и dataloaders
    train_dataset = AdvancedChessDataset(train_df, tokenizer, move_vocab)
    val_dataset = AdvancedChessDataset(val_df, tokenizer, move_vocab)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Создание модели
    model = ImprovedChessModel(input_size, vocab_size, hidden_size=1024)

    print(f"\nМодель создана")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Архитектура: {model}")

    # Обучение
    train_losses, val_accuracies, best_accuracy = advanced_train_model(
        model, train_loader, val_loader, epochs=30
    )

    # Загрузка лучшей модели для оценки
    checkpoint = torch.load('best_chess_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("="*60)
    print(f"Лучшая точность на валидации: {best_accuracy:.4f}")

    # Расчет всех метрик
    top1_acc = calculate_top_k_accuracy(model, val_loader, k=1)
    top3_acc = calculate_top_k_accuracy(model, val_loader, k=3)
    top5_acc = calculate_top_k_accuracy(model, val_loader, k=5)
    perplexity = calculate_perplexity(model, val_loader)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    # Анализ предсказаний
    analyze_predictions(model, val_loader, move_vocab, tokenizer, n_samples=5)

    # Визуализация обучения
    plot_training_history(train_losses, val_accuracies)

    # Сохранение словаря и токенизатора для будущего использования
    torch.save({
        'move_vocab': move_vocab,
        'tokenizer': tokenizer,
        'input_size': input_size,
        'vocab_size': vocab_size
    }, 'model_assets.pth')

    print(f"\nМодель и ассеты сохранены для будущего использования!")

# ==================== ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ====================

def predict_moves(fen_position, model_path='best_chess_model.pth', assets_path='model_assets.pth', top_k=5):
    """Функция для предсказания ходов по FEN позиции"""
    # Загрузка ассетов
    assets = torch.load(assets_path)
    move_vocab = assets['move_vocab']
    tokenizer = assets['tokenizer']
    input_size = assets['input_size']
    vocab_size = assets['vocab_size']

    # Загрузка модели
    model = ImprovedChessModel(input_size, vocab_size)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Преобразование FEN и предсказание
    input_tensor = tokenizer.fen_to_advanced_tensor(fen_position)
    input_tensor = torch.FloatTensor(input_tensor).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        topk_probs, topk_indices = torch.topk(probabilities, top_k, dim=1)

    # Форматирование результатов
    moves = []
    for i in range(top_k):
        move = move_vocab.index_to_move(topk_indices[0][i].item())
        prob = topk_probs[0][i].item()
        moves.append((move, prob))

    return moves

if __name__ == "__main__":
    main()