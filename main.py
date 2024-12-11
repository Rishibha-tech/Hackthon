from src.preprocessing import preprocess_images
from src.training import train_model, save_model
from src.evaluation import evaluate_model
import os

def main():
    # Paths for data
    data_dir = 'data'
    processed_dir = os.path.join(data_dir, 'processed')
    raw_dir = os.path.join(data_dir, 'raw')
    model_dir = 'models'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Preprocessing
    print("Preprocessing images...")
    preprocess_images(input_dir=raw_dir,
                      output_dir=processed_dir)



    # Training
    print("Training the model...")
    model = train_model(raw_dir, processed_dir)
    model_path = os.path.join(model_dir, 'signature_model.h5')
    save_model(model, model_path)

    # Evaluation
    print("Evaluating the model...")
    evaluate_model(model_path, processed_dir)

if __name__ == "__main__":
    main()
