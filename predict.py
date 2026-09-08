import torch
import pandas as pd

from data import load_data, create_loader_test
from model import load

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_submission_csv(model, dataloader):
    model.eval()
    model.to(device)

    image_ids = []
    labels = []

    img_id = 1

    with torch.no_grad():
        for (X,) in dataloader:
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)

            batch_size = preds.size(0)
            image_ids.extend(range(img_id, img_id + batch_size))
            labels.extend(preds.cpu().tolist())

            img_id += batch_size

    df = pd.DataFrame({
        "ImageID": image_ids,
        "Label": labels
    })

    output_path = "results/submit-020526.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")

def main():
    model = load("results/model.safetensors")
    test_X = load_data(test_only=True)
    dataloader = create_loader_test(test_X)
    generate_submission_csv(model, dataloader)

if __name__ == "__main__":
    main()