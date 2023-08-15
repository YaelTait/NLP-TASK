import pandas as pd
from data.metric_variations import metric_variations
import re
import spacy
from fuzzywuzzy import fuzz
import csv


class MetricFinder:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # read the questions:
        self.questions = self._load_questions()
        # Processing of variations of the metrics:
        self.processed_variations = self._preprocess_variations()
        # Create a file for the results:
        self._creat_the_results_file()
        # A high ans low thresholds for the similarity of a question
        # to a variation of a metric:
        self.high_threshold = 0.8
        self.low_threshold = 0.7
        self.maximum_metric_length = 4

    def _preprocess_variations(self):
        processed_variations = {}
        for key, variations in metric_variations.items():
            processed_variations[key] = [
                self._preprocess_text(variation) for variation in variations
            ]
        return processed_variations

    def _load_questions(self):
        data = pd.read_csv("./data/Dataset for Yael.csv")
        return data["question"].tolist()

    def _creat_the_results_file(self):
        # Write data to the CSV file
        with open("./data/results.csv", "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header row
            csv_writer.writerow(["Question", "Similarity", "Metric"])

    def _write_the_result(self, result_to_insert):
        with open("./data/results.csv", "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(result_to_insert)

    # Calculating the similarity between a part of a question and a matric,
    # using Levenshtein distance algorithem
    def _compute_similarity(self, part, variation):
        similarity = fuzz.ratio(part, variation) / 100
        return similarity

    # # Preprocess function to clean  text
    def _preprocess_text(self, text):
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters and punctuation
        text = text.lower()  # Convert to lowercase

        # Process the text with spaCy
        doc = self.nlp(text)

        # Remove stop words and lemmatization
        cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop]

        # Join words back into a cleaned text
        cleaned_text = " ".join(cleaned_tokens)

        return cleaned_text

    def _find_metrics_in_questions(self):
        for question in self.questions:
            # Processing the question
            question = self._preprocess_text(question)
            doc_question = self.nlp(question)
            doc_len = len(doc_question)
            max_similarity = 0.0
            metric = None
            should_continue_checking = True  # Set the flag to True
            # Running over all possible lengths of a metric
            for length in range(self.maximum_metric_length, 0, -1):
                # Comparison of possible lengths of a matrix, from each possible start in the question
                for start in range(doc_len - length + 1):
                    end = start + length
                    part_tokens = doc_question[start:end]
                    part = " ".join(token.text for token in part_tokens)
                    # Comparison of each part for each of the variations:
                    for metric, variations in self.processed_variations.items():
                        for variation in variations:
                            similarity = self._compute_similarity(part, variation)
                            # If a high similarity is found -
                            # save the result and move on to the next question
                            if similarity > self.high_threshold:
                                self._write_the_result([question, similarity, metric])
                                # The flag means there is no need to continue searching for this question
                                should_continue_checking = False  # Set the flag to True
                                break  # This will break the inner loop over variations
                            # save the most similar metric so far:
                            if similarity > max_similarity:
                                max_similarity = similarity
                                metric = metric
                        if not should_continue_checking:
                            break  # This will break the outer loop over variations
                    if not should_continue_checking:
                        break  # This will break the outer loop over diffrents starts

                if not should_continue_checking:
                    break  # This will break the outer loop over diffrents lengths
            # If the found metric exceeds the low threshold,
            # and no metric has already been found that exceeds the high threshold
            # save the results:
            if max_similarity > self.low_threshold and should_continue_checking:
                self._write_the_result([question, max_similarity, metric])


if __name__ == "__main__":
    metric_finder = MetricFinder()
    metric_finder._find_metrics_in_questions()
