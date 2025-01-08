export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

cd longbench
FLEXGEN_PATH=$PWD/../../flexgen
SCHEME="infinigen"

rm $FLEXGEN_PATH/flexgen/flex_opt.py
rm $FLEXGEN_PATH/flexgen/flex_llama.py
rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
rm $FLEXGEN_PATH/flexgen/run_longbench.py
rm $FLEXGEN_PATH/flexgen/run_arxivsum.py
ln -s ../$SCHEME/flex_opt.py $FLEXGEN_PATH/flexgen/flex_opt.py
ln -s ../$SCHEME/flex_llama.py $FLEXGEN_PATH/flexgen/flex_llama.py
ln -s ../$SCHEME/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py
ln -s ../$SCHEME/run_longbench.py $FLEXGEN_PATH/flexgen/run_longbench.py
ln -s ../$SCHEME/run_arxivsum.py $FLEXGEN_PATH/flexgen/run_arxivsum.py

llama=meta-llama/Llama-2-7b-chat-hf
lwm=LargeWorldModel/LWM-Text-Chat-1M
DS1=qasper
# ["multifieldqa_en", "qasper", "gov_report", "qmsum", "samsum"]

# for DS in "qasper"; do
#     for alpha in 56; do
#         python -m flexgen.run_longbench \
#             --model $lwm \
#             --path ~/infinigen-data/llama_weights/ \
#             --overlap false \
#             --percent 100 0 0 100 100 0 \
#             --gpu-batch-size 1 \
#             --num-gpu-batches 1 \
#             --warmup-input-path pg19_firstbook.txt \
#             --alpha $alpha \
#             --partial-weight-ratio 0.3 \
#             --max-num-kv 32000 \
#             --dataset-name $DS
#     done
# done

# for DS in "qasper" "gov_report" "qmsum" "vcsum" "multi_news"; do
for DS in "qmsum"; do
    for alpha in 25 35 50 60; do
        python -m flexgen.run_longbench \
            --model $lwm \
            --path ~/infinigen-data/llama_weights/ \
            --overlap false \
            --percent 100 0 0 100 100 0 \
            --gpu-batch-size 1 \
            --num-gpu-batches 1 \
            --warmup-input-path pg19_firstbook.txt \
            --alpha $alpha \
            --partial-weight-ratio 0.3 \
            --max-num-kv 32000 \
            --dataset-name $DS
    done
done

for DS in "multi_news"; do
    for alpha in 50 62 75 95; do
        python -m flexgen.run_longbench \
            --model $lwm \
            --path ~/infinigen-data/llama_weights/ \
            --overlap false \
            --percent 100 0 0 100 100 0 \
            --gpu-batch-size 1 \
            --num-gpu-batches 1 \
            --warmup-input-path pg19_firstbook.txt \
            --alpha $alpha \
            --partial-weight-ratio 0.3 \
            --max-num-kv 32000 \
            --dataset-name $DS
    done
done

# for alpha in 35 50 62 80 100; do
#     python -m flexgen.run_arxivsum \
#         --model $lwm \
#         --path ~/infinigen-data/llama_weights/ \
#         --overlap false \
#         --percent 100 0 0 100 100 0 \
#         --gpu-batch-size 1 \
#         --num-gpu-batches 1 \
#         --warmup-input-path pg19_firstbook.txt \
#         --alpha $alpha \
#         --partial-weight-ratio 0.3 \
#         --max-num-kv 32000 \
# done