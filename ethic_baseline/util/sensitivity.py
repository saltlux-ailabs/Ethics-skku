import numpy
import json
from tqdm import tqdm

def calcuate_gender_sensitivity(model, tokenizer, cls_type='binary', lang='ko'):

    model.eval()
    model.cuda()

    template_list = None
    gender_group = None

    if lang == "ko":
        with open("../data/sensitivity/sensitiity_ko.txt") as f:
            template_list = f.readlines()
    else:
        NotImplementedError

    with open("../data/senstivity/gender_group.json") as f:
        gender_group_list = json.load(f)

    sensitivity_score = []

    for template_i in tqdm(template_list):
        m_score = 0
        f_score = 0

        for target_i in gender_group_list['M']:
            model_input = tokenizer(template_i.replace("<group>", target_i))
            model_input = {k: v.cuda() for k, v in model_input.items()}
            model_output = model(**model_input)

            if cls_type == 'binary':
                pass
            else:
                model

        for target_i in gender_group_list['M']:
            model_input = tokenizer(template_i.replace("<group>", target_i))
            model_input = {k: v.cuda() for k, v in model_input.items()}
            model_output = model(**model_input)

            print(model_output.shape())

            break

            if cls_type == 'binary':
                pass
            else:
                model