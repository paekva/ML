import matplotlib.pyplot as plt


def draw_graphics(H):
    # Получение ошибки и точности в процессе обучения
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    mae = H.history['mae']
    val_mae = H.history['val_mae']
    epochs = range(1, len(loss) + 1)

    # Построение графика ошибки
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Построение графика точности
    plt.clf()
    plt.plot(epochs, mae, 'bo', label='Training mae')
    plt.plot(epochs, val_mae, 'b', label='Validation mae')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('mae')
    plt.legend()
    plt.show()
