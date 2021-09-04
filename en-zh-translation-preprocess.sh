#!/bin/sh
src=zh
tgt=en

SCRIPTS=./mosesdecoder/scripts
TOKENIZER=${SCRIPTS}/tokenizer/tokenizer.perl
DETOKENIZER=${SCRIPTS}/tokenizer/detokenizer.perl
LC=${SCRIPTS}/tokenizer/lowercase.perl
TRAIN_TC=${SCRIPTS}/recaser/train-truecaser.perl
TC=${SCRIPTS}/recaser/truecase.perl
DETC=${SCRIPTS}/recaser/detruecase.perl
NORM_PUNC=${SCRIPTS}/tokenizer/normalize-punctuation.perl
CLEAN=${SCRIPTS}/training/clean-corpus-n.perl
BPEROOT=./subword-nmt/subword_nmt
MULTI_BLEU=${SCRIPTS}/generic/multi-bleu.perl
MTEVAL_V14=${SCRIPTS}/generic/mteval-v14.pl

data_dir=./nmt/data/v15news
model_dir=./nmt/models/v15news
utils=./nmt/utils

# step 1 generate raw.zh, raw.en
echo "step 1, generate raw file"
python ${utils}/cut2.py ${data_dir}/news-commentary-v15.en-zh.tsv ${data_dir}/

# step 2 generate norm.zh, norm.en
echo "step 2, generate norm file"
perl ${NORM_PUNC} -l en < ${data_dir}/raw.en > ${data_dir}/norm.en
perl ${NORM_PUNC} -l zh < ${data_dir}/raw.zh > ${data_dir}/norm.zh

# step 3 generate norm.seg.zh
echo "step 3, apply jieba to norm.zh"
python -m jieba -d " " ${data_dir}/norm.zh > ${data_dir}/norm.seg.zh

# step 4 generate tok file, 目的是： 1.将英文单词与标点符号用空格分开 2.将多个连续空格简化为一个空格 3.将很多符号替换成转义字符，如：把"替换成&quot;、把can't替换成can &apos;t 
echo "step 4, ..."
${TOKENIZER} -l en < ${data_dir}/norm.en > ${data_dir}/norm.tok.en
${TOKENIZER} -l zh < ${data_dir}/norm.seg.zh > ${data_dir}/norm.seg.tok.zh

# step 5 truecase, 目的是： 对上述处理后的英文文件(norm.tok.en)进行大小写转换处理(对于句中的每个英文单词，尤其是句首单词，在数据中学习最适合它们的大小写形式
echo "step 5, 对英文语料大小写转换"
${TRAIN_TC} --model ${model_dir}/truecase-model.en --corpus ${data_dir}/norm.tok.en
${TC} --model ${model_dir}/truecase-model.en < ${data_dir}/norm.tok.en > ${data_dir}/norm.tok.true.en

# step 6 bpe, generate bpe token file, 目的是： 对上述处理后的双语文件(norm.tok.true.en, norm.seg.tok.zh)进行子词处理(可以理解为更细粒度的分词)
echo "step 6, subword process"
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.tok.true.en  -s 32000 -o ${model_dir}/bpecode.en --write-vocabulary ${model_dir}/voc.en
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.en --vocabulary ${model_dir}/voc.en < ${data_dir}/norm.tok.true.en > ${data_dir}/norm.tok.true.bpe.en

python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/norm.seg.tok.zh  -s 32000 -o ${model_dir}/bpecode.zh --write-vocabulary ${model_dir}/voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/norm.seg.tok.zh > ${data_dir}/norm.seg.tok.bpe.zh

# step 7 clean, 目的是：对上述处理后的双语文件(norm.tok.true.bpe.en, norm.seg.tok.bpe.zh)进行过滤(可以过滤最小长度和最大长度之间的句对，这样能够有效过滤空白行。还可以过滤长度比不合理的句对)
echo "step 7, clean"
cp ${data_dir}/norm.seg.tok.bpe.zh ${data_dir}/toclean.zh
cp ${data_dir}/norm.tok.true.bpe.en ${data_dir}/toclean.en
${CLEAN} ${data_dir}/toclean zh en ${data_dir}/clean 1 256

# step 8, split, 划分训练集、测试机、验证集
python ${utils}/split.py ${data_dir}/clean.zh ${data_dir}/clean.en ${data_dir}/

# step 9 generate data-bin
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${data_dir}/train --validpref ${data_dir}/valid --testpref ${data_dir}/test \
    --destdir ${data_dir}/data-bin
    
# step 10 start training
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fairseq-train ${data_dir}/data-bin --arch transformer \
	--source-lang ${src} --target-lang ${tgt}  \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --num-workers 8 \
	--save-dir ${model_dir}/checkpoints &

# step 11 test
fairseq-generate ${data_dir}/data-bin \
    --path ${model_dir}/checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 8 > ${data_dir}/result/bestbeam8.txt
  
# step 12 evaluate
grep ^H ${data_dir}/result/bestbeam8.txt | cut -f3- > ${data_dir}/result/predict.tok.true.bpe.en
grep ^T ${data_dir}/result/bestbeam8.txt | cut -f2- > ${data_dir}/result/answer.tok.true.bpe.en
## 有两种方法可以去除bpe符号，第一种是在解码时添加--remove-bpe参数，第二种是使用sed指令：
sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/result/predict.tok.true.bpe.en  > ${data_dir}/result/predict.tok.true.en
sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/result/answer.tok.true.bpe.en  > ${data_dir}/result/answer.tok.true.en
## 需要使用detruecase.perl将文件中的大小写恢复正常：
${DETC} < ${data_dir}/result/predict.tok.true.en > ${data_dir}/result/predict.tok.en
${DETC} < ${data_dir}/result/answer.tok.true.en > ${data_dir}/result/answer.tok.en
## bleu score
${MULTI_BLEU} -lc ${data_dir}/result/answer.tok.en < ${data_dir}/result/predict.tok.en
## 最后一步，是使用detokenize.perl得到纯预测文本
${DETOKENIZER} -l en < ${data_dir}/result/predict.tok.en > ${data_dir}/result/predict.en
