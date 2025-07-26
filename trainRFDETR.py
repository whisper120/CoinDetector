from rfdetr import RFDETRBase

def train():
    model = RFDETRBase()

    model.train(
        dataset_dir="train",    # your dataset folder
        epochs=75,
        batch_size=8,
        lr=1e-4,
        resume="third_50epochs.pth"
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    train()
