import click
from torch import jit, from_numpy, softmax, argmax, no_grad
import numpy as np
from utils.calculate_features import create_feature_sample

LABEL2EMO = {0: "angry", 1: "sad", 2: "neutral", 3: "positive", 4: "noise"}

@click.command()
@click.option("--file_path", help=".wav file location")
@click.option("--model_path", help="traced model file location")
def main(file_path, model_path) -> None:
    np.seterr(divide="ignore")
    test_sample = create_feature_sample(file_path)[None]
    model = jit.load(model_path)
    model.eval()
    with no_grad():
        logits = model(from_numpy(test_sample[None, :])).reshape(-1)
        prediction = softmax(logits, dim=-1)
        output_label = argmax(prediction).item()
        print(f"{LABEL2EMO[output_label]} probability: {prediction[output_label]:.4f} ({prediction.tolist()})")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
