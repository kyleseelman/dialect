import logging
import pathlib
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional, Union
import dspy
from dspy.datasets import Dataset
import pathlib
from sklearn.model_selection import train_test_split
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate

class SyntheticDataset(Dataset):
    def __init__(
        self,
        data_fpath: Union[pathlib.Path, str],
        dev_size: Optional[float] = 0.2,
        test_size: Optional[float] = 0.2,
        input_keys: str = ["TOPIC", "INPUT_SENTENCE"],
        seed: Optional[int] = 11235,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self._train = []
        self._dev = []
        self._test = []

        train_data = pd.read_excel(data_fpath).dropna()

        train_data, temp_data = train_test_split(
            train_data, test_size=dev_size + test_size, random_state=seed)
        dev_data, test_data = train_test_split(
            temp_data, test_size=test_size / (dev_size + test_size), random_state=seed)

        self._train = [
            dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(train_data)
        ]
        self._dev = [
            dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(dev_data)
        ]
        self._test = [
            dspy.Example({**row}).with_inputs(*input_keys) for row in self._convert_to_json(test_data)
        ]

    def _convert_to_json(self, data: pd.DataFrame):
        if data is not None:
            return data.to_dict(orient='records')
        

class Transform(dspy.Signature):
    """
    Rewrite the sentence by replacing its key elements (e.g., objects, subjects, actions) to reflect the given topic, while preserving the original grammatical structure.
    The rewritten sentence should align with the meaning and focus of the given topic.

    ----------------------------------------------------------------------------
    Examples
    ----------------------------------------------------------------------------
    TOPIC: Sports
    INPUT_SENTENCE: I made a master after my bachelor because I wanted to pursue an academic path.
    TARGET_SENTENCE: I enrolled in the sports program in high school because I’m passionate about taekwondo.

    TOPIC: Food
    INPUT_SENTENCE: Three in ten South Africans are younger than 15, meaning that they did not live a day under apartheid.
    TARGET_SENTENCE: Three in ten South Africans prefer traditional dishes like bobotie, meaning they grew up enjoying local flavors rather than fast food.

    TOPIC: Maternal Healthcare
    INPUT_SENTENCE: Diamond apparently believed his own rhetoric – that he and his bank are critical to economic prosperity in the UK.
    TARGET_SENTENCE: Diamond apparently believed his own rhetoric – that he and his clinic are critical to improving maternal healthcare in the UK.

    TOPIC: SPACE TOURISM
    INPUT_SENTENCE: We should adopt policies that benefit the economy in the short run at reasonable long-run cost, and reject those that do not.
    TARGET_SENTENCE: We should adopt policies that promote space tourism in the short run while ensuring sustainable development for future space exploration, and reject those that do not.

    TOPIC: QUANTUM COMPUTING
    INPUT_SENTENCE: Their achievement remains one of the greatest in recent history.
    TARGET_SENTENCE: Their breakthrough in quantum computing remains one of the greatest technological advancements in recent history.
    ----------------------------------------------------------------------------

    Important: The TARGET_SENTENCE should be different from the INPUT_SENTENCE.
    """

    TOPIC = dspy.InputField()
    INPUT_SENTENCE = dspy.InputField()
    TARGET_SENTENCE = dspy.OutputField()        

class Transform(dspy.Signature):
    """
    Rewrite the sentence by replacing its key elements (e.g., objects, subjects, actions) to reflect the given topic, while preserving the original grammatical structure.
    The rewritten sentence should align with the meaning and focus of the given topic.
    """

    TOPIC = dspy.InputField(desc="The subject or theme that the TARGET_SENTENCE should reflect.")
    INPUT_SENTENCE = dspy.InputField(desc="The original sentence that is to be modified.")
    TARGET_SENTENCE = dspy.OutputField(desc="The modified sentence that should incorporate the given TOPIC.")

class TransformModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.transform = dspy.Predict(Transform)

    def forward(self, TOPIC, INPUT_SENTENCE):
        out = self.transform(TOPIC=TOPIC, INPUT_SENTENCE=INPUT_SENTENCE).TARGET_SENTENCE

        if "TARGET SENTENCE: " in out:
            out = out.split("TARGET SENTENCE: ")[1]
            
        return dspy.Prediction(target = out)

class Judge(dspy.Signature):
    """Judge whether the TARGET_SENTENCE is a modification of the INPUT_SENTENCE that appropriately incorporates the given TOPIC, while maintaining a similar structure."""

    TOPIC = dspy.InputField(desc="The subject or theme that the TARGET_SENTENCE should reflect.")
    INPUT_SENTENCE = dspy.InputField(desc="The original sentence that is to be modified.")
    TARGET_SENTENCE = dspy.InputField(desc="The modified sentence that should incorporate the given TOPIC.")
    correct = dspy.OutputField(desc="A binary output indicating whether the TARGET_SENTENCE is a valid and accurate modification of the INPUT_SENTENCE according to the TOPIC.", prefix="Correct[Yes/No]:")
    
class EnEsTranslator(dspy.Module):
    
    def __init__(self):
        super().__init__()
        self.translate = dspy.ChainOfThought("English -> Spanish")
    
    def forward(self, in_sentence):
        return self.translate(English=in_sentence)
    
class DatasetGenerator(object):
    def __init__(
        self,
        model_type: str = "llama",
        open_ai_model: str = "gpt-3.5-turbo",
        path_open_api_key="/export/usuarios_ml4ds/lbartolome/thuban/crispy-robot/.env",
        do_train: bool = False,
        data_fpath: Union[pathlib.Path, str] = pathlib.Path(__file__).parent.joinpath("data/tr_data/dialect_dtset.xlsx"),
        trained_prompt: str = pathlib.Path(
            __file__).parent.joinpath("data/prompts/DatasetGenerator.json"),
    ) -> None:
        
        # redirect dspy logs
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._logger.info("Logger initialized.")
        
        if model_type == "llama":
            self.lm = dspy.HFClientTGI(
                model="meta-llama/Meta-Llama-3-8B",
                port=8090, url="http://127.0.0.1"
            )
        elif model_type == "openai":
            load_dotenv(path_open_api_key)
            api_key = os.getenv("OPENAI_API_KEY")
            os.environ["OPENAI_API_KEY"] = api_key
            self.lm = dspy.OpenAI(model=open_ai_model)
            
        dspy.settings.configure(lm=self.lm)
            
        if not do_train:
            if not pathlib.Path(trained_prompt).exists():
                self._logger.error(f"-- -- Trained prompt file not found at {trained_prompt}.")
                raise FileNotFoundError(f"Trained prompt file not found at {trained_prompt}.")
            self.module = TransformModule()
            self.module.load(trained_prompt)
            self._logger.info(f"-- -- Trained prompt loaded from {trained_prompt}.")
        else:
            if not pathlib.Path(data_fpath).exists():
                self._logger.error(f"-- -- Data file not found at {data_fpath}.")
                raise FileNotFoundError(f"Data file not found at {data_fpath}.")
            
            self.gpt4T = dspy.OpenAI(
                model="gpt-4o",
                max_tokens=1000, 
                model_type="chat")
            self.judge = dspy.ChainOfThought(Judge)
            self._train_module(data_fpath, trained_prompt)
        
        self.en_es_translator = EnEsTranslator()
            
    def _train_module(
        self,
        data_fpath: Union[pathlib.Path, str],
        trained_prompt: Union[pathlib.Path, str]
    ) -> None:
        """
        Trains the TransformModule using the given dataset ans saves the trained prompt to the specified file.
        
        Parameters
        ----------
        data_fpath : Union[pathlib.Path, str]
            The path to the dataset file.
        trained_prompt : Union[pathlib.Path, str]
            The path to save the trained prompt.
        """
        
        self._logger.info("-- -- Training the TransformModule.")
        self.module = self.optimize_module(data_fpath)
        self._logger.info("-- -- Training completed.")
        self.module.save(trained_prompt)
        self._logger.info(f"-- -- Trained prompt saved to {trained_prompt}.")
        
    def optimize_module(
        self,
        data_path,
        mbd=4,
        mld=16,
        ncp=2,
        mr=1,
        dev_size=0.25 
    ) -> TransformModule:
        """
        Optimizes the TransformModule using the given dataset.
        
        Parameters
        ----------
        data_fpath : Union[pathlib.Path, str]
            The path to the dataset file.
        mbd : int
            Maximum number of bootstrapped demonstrations per predictor
        mld : int
            Maximum number of labeled demonstrations per predictor
        ncp : int
            Number of candidate programs to generate during random search
        mr : int
            Maximum number of bootstrapping rounds
        dev_size : float
            The fraction of the dataset to use for the validation set.
        
        Returns
        -------
        TransformModule
            The optimized TransformModule.
        """
        
        self._logger.info("-- -- Optimizing the TransformModule.")
        
        dataset = SyntheticDataset(data_path, dev_size=dev_size)
        self._logger.info(f"-- -- Dataset loaded from {data_path}.")
        
        trainset = dataset._train
        devset = dataset._dev
        testset = dataset._test
        self._logger.info(f"-- -- Dataset split into train, dev, and test sets.")

        config = dict(max_bootstrapped_demos=mbd, max_labeled_demos=mld,
                      num_candidate_programs=ncp, max_rounds=mr)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.synthetic_score, **config)
        
        compiled_pred = teleprompter.compile(
            TransformModule(), trainset=trainset, valset=devset)

        self._logger.info(f"-- -- Module compiled. Evaluating on test set...")
        
            # Apply on test set
        tests = []
        for el in testset:
            output = compiled_pred(el.TOPIC,el.INPUT_SENTENCE)
            tests.append([el.TOPIC, el.INPUT_SENTENCE, el.TARGET_SENTENCE,
                        output["target"], self.synthetic_score(el, output)])

            df = pd.DataFrame(
                tests, columns=["TOPIC", "INPUT_SENTENCE", "GR_TARGET_SENTENCE", "PRED_TARGET_SENTENCE", "METRIC"])

            print(df)

            evaluate = Evaluate(
                devset=devset, metric=self.synthetic_score, num_threads=1, display_progress=True)
            compiled_score = evaluate(compiled_pred)
            uncompiled_score = evaluate(TransformModule())

            print(
                f"## TransformModule Score for uncompiled: {uncompiled_score}")
            print(
                f"## TransformModule Score for compiled: {compiled_score}")
            print(f"Compilation Improvement: {compiled_score - uncompiled_score}%")

            return compiled_pred
    
    def synthetic_score(self, example, pred, trace=None):
        """
        Scores the prediction based on the correctness of the transformation.
        
        Parameters
        ----------
        example : dspy.Example
            The example containing the input sentence and the target sentence.
        pred : dspy.Prediction
        
        Returns
        -------
        int
            The score based on the correctness of the transformation (1 if correct, 0 otherwise).
        """
        with dspy.context(lm=self.gpt4T):
            correct = self.judge(TOPIC=example.TOPIC, INPUT_SENTENCE=example.INPUT_SENTENCE, TARGET_SENTENCE=pred.target).correct
            print(correct)
        return int("Yes" in correct)
    
    def generate_dataset(
        self,
        df,
        column_apply: str = "input_en",
        topics = pathlib.Path(__file__).parent.joinpath("data/topics/topics_generic.txt"),
        sample=100
        ):
        """
        Generates a synthetic dataset by transforming the given sentences to reflect the specified topics.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the sentences to be transformed.
        column_apply : str
            The column in the DataFrame containing the sentences to be transformed.
        topics : pathlib.Path
            The path to the file containing the topics to be applied to the sentences.
        sample : int
            The number of sentences to sample from the DataFrame for transformation.
        
        Returns
        -------
        pd.DataFrame
            The synthetic dataset containing the transformed sentences, with columns for the topic, input sentence, target sentence, and translated target sentence.
        """
        
        with open(topics, "r") as f:
            topics = f.readlines()
        topics = [topic.strip() for topic in topics]
        
        generated = [] 
        for id_, el in df.sample(n=sample).iterrows():
            
            for topic in topics:
                print(f"-- -- Generating for topic: {topic}")
                output_sent = self.module(topic, el[column_apply]).target
                if el[column_apply] != output_sent:
                    output_sent_tr = self.en_es_translator(output_sent).Spanish
                    # sometimes the generated sentence is the same as the input sentence, so we skip those
                    generated.append([topic, el[column_apply], output_sent, output_sent_tr])
        final_df = pd.DataFrame(generated, columns=["TOPIC", "INPUT_SENTENCE", "TARGET_SENTENCE", "TARGET_SENTENCE_TR"])
        # Remove duplicates based on TARGET_SENTENCE, keep first
        final_df = final_df.drop_duplicates(subset=["TARGET_SENTENCE"])
        return final_df
            
            