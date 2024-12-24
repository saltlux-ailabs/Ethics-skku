### Binary_Kor
batch_size=32
path="/nlp_data/yumin/data/total/multi/"
for model in "klue/bert-base"; do
    for mode in "problems" "topic"; do
        for hyper_param in "3e-5,0.1,0.01"; do
            echo "*&*&*&*&*&*&*&*&%%%${model}%%%${mode}%%%${hyper_param}%%%"
            echo ${data}
            CUDA_VISIBLE_DEVICES=2 python3 train_multi.py --model "${model}" --hyper_param ${hyper_param} --batch_size ${batch_size} --num_epochs 5 --train_data "${path}${mode}/train.json" --valid_data "${path}${mode}/test.json"  --test_data "${path}${mode}/test.json" --mode ${mode}
        done
    done
done
# nohup ./train2_multi.sh > ethic_multi-bert.out & nohup ./train3_multi.sh > ethic_multi-roberta.out &