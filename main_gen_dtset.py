import time
import pandas as pd
from src.dataset_generation.dataset_generator import DatasetGenerator
from datasets import load_dataset

def main():

    ds = load_dataset("Helsinki-NLP/news_commentary", "en-es")
    train_df = pd.DataFrame(ds['train'])
    train_df["input_en"] = train_df["translation"].apply(lambda x: x["en"])
    train_df["input_es"] = train_df["translation"].apply(lambda x: x["es"])
    train_df = train_df[['id', 'input_en', 'input_es']]
        
    # Create dataset_generator object
    dataset_generator = DatasetGenerator(
        do_train=True
    )
    
    time_start = time.time()
    # Generate dataset
    df_out = dataset_generator.generate_dataset(
        df = train_df,
        column_apply = "input_en",
        sample=len(train_df)
    )
    
    print("Time taken: ", time.time() - time_start)
    
    df_out.to_excel("data/out/dialect_synthetic.xlsx", index=False)
    
    print("Dataset generated and saved to data/out/dialect_synthetic.xlsx")
    
if __name__ == "__main__":
    main()