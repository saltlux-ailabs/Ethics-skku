### Binary_Kor
batch_size=32
path="/nlp_data/yumin/data/total/multi/"
for model in "klue/bert-base"; do
    for mode in "problems"; do # "topic"
        for hyper_param in "3e-5,0.1,0.01"; do
            for reg_strength in "0.01"; do
                CUDA_VISIBLE_DEVICES=3 python3 train_entropy.py --model "${model}" \
                --hyper_param ${hyper_param} --batch_size ${batch_size} --num_epochs 10 --train_data "${path}${mode}/train.json" \
                --valid_data "${path}${mode}/test.json" --test_data "${path}${mode}/test.json" --mode ${mode} \
                --reg_strength ${reg_strength} --save_dir "/nlp_data/yumin/ethic_baseline/models/multi_debias_attention"
            done
        done    
    done
done