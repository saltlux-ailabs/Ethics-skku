### Binary_Kor
batch_size=32
path="/nlp_data/yumin/data/total/binary/"
for model in "klue/bert-base"; do
    for hyper_param in "3e-5,0.1,0.01"; do
        echo "*&*&*&*&*&*&*&*&"
        echo ${model}
        echo ${hyper_param}
        echo ${data}
        CUDA_VISIBLE_DEVICES=2 python3 train_bin.py --num_labels 2 --model "${model}" --hyper_param ${hyper_param} --batch_size ${batch_size} --num_epochs 5 --train_data "${path}train.json" --valid_data "${path}test.json"  --test_data "${path}test.json"
    done
done
# nohup ./train2_bin.sh > ethic_bin-bert.out & nohup ./train3_bin.sh > ethic_bin-roberta.out &