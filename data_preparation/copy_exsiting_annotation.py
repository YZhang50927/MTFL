import re

input_file = "H:\\Projects\\VAD\\Datasets\\baseline\\UCF\\Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
output_file = "H:\\Projects\\VAD\\workspace\\data_preparation\\annotation\\doc\\new annotation.txt"
keyword_file = "H:\\Projects\\VAD\\workspace\\data_preparation\\reclassification\\BodyExposedAccidentsInUCF.txt"
original_word = "RoadAccidents"
replace_word = "BodyExposedAccidents"

if __name__ == "__main__":
    with open(keyword_file, 'r') as f_keyword:
        keywords = [line.strip('\n') for line in f_keyword]

    with open(input_file, 'r') as f_input, open(output_file, 'a') as f_output:
        for line in f_input:
            words = re.findall(r'\b\w+\b', line)
            for keyword in keywords:
                if keyword in line:
                    for word in words:
                        if word == original_word:
                            line = line.replace(word, replace_word)
                            f_output.write(line)
                            break

