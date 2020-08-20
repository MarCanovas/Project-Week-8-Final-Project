echo "Testing 8M..."

python evaluate.py 8M fasttext lstm
python evaluate.py 8M fasttext cnn
python evaluate.py 8M fasttext bi

python evaluate.py 8M glove lstm
python evaluate.py 8M glove cnn
python evaluate.py 8M glove bi

python evaluate.py 8M wiki lstm
python evaluate.py 8M wiki cnn
python evaluate.py 8M wiki bi

python evaluate.py 8M embeddings-l-model lstm
python evaluate.py 8M embeddings-l-model cnn
python evaluate.py 8M embeddings-l-model bi

echo "Testing discredit..."

python evaluate.py discredit fasttext lstm
python evaluate.py discredit fasttext cnn
python evaluate.py discredit fasttext bi

python evaluate.py discredit glove lstm
python evaluate.py discredit glove cnn
python evaluate.py discredit glove bi

python evaluate.py discredit wiki lstm
python evaluate.py discredit wiki cnn
python evaluate.py discredit wiki bi

python evaluate.py discredit embeddings-l-model lstm
python evaluate.py discredit embeddings-l-model cnn
python evaluate.py discredit embeddings-l-model bi


echo "Testing ecuador..."

python evaluate.py ecuador fasttext lstm
python evaluate.py ecuador fasttext cnn
python evaluate.py ecuador fasttext bi

python evaluate.py ecuador glove lstm
python evaluate.py ecuador glove cnn
python evaluate.py ecuador glove bi

python evaluate.py ecuador wiki lstm
python evaluate.py ecuador wiki cnn
python evaluate.py ecuador wiki bi

python evaluate.py ecuador embeddings-l-model lstm
python evaluate.py ecuador embeddings-l-model cnn
python evaluate.py ecuador embeddings-l-model bi

echo "Testing feminism..."

python evaluate.py feminism fasttext lstm
python evaluate.py feminism fasttext cnn
python evaluate.py feminism fasttext bi

python evaluate.py feminism glove lstm
python evaluate.py feminism glove cnn
python evaluate.py feminism glove bi

python evaluate.py feminism wiki lstm
python evaluate.py feminism wiki cnn
python evaluate.py feminism wiki bi

python evaluate.py feminism embeddings-l-model lstm
python evaluate.py feminism embeddings-l-model cnn
python evaluate.py feminism embeddings-l-model bi

echo "Testing feminist..."

python evaluate.py feminists fasttext lstm
python evaluate.py feminists fasttext cnn
python evaluate.py feminists fasttext bi

python evaluate.py feminists glove lstm
python evaluate.py feminists glove cnn
python evaluate.py feminists glove bi

python evaluate.py feminists wiki lstm
python evaluate.py feminists wiki cnn
python evaluate.py feminists wiki bi

python evaluate.py feminists embeddings-l-model lstm
python evaluate.py feminists embeddings-l-model cnn
python evaluate.py feminists embeddings-l-model bi

echo "Testing machismo..."

python evaluate.py machismo fasttext lstm
python evaluate.py machismo fasttext cnn
python evaluate.py machismo fasttext bi

python evaluate.py machismo glove lstm
python evaluate.py machismo glove cnn
python evaluate.py machismo glove bi

python evaluate.py machismo wiki lstm
python evaluate.py machismo wiki cnn
python evaluate.py machismo wiki bi

python evaluate.py machismo embeddings-l-model lstm
python evaluate.py machismo embeddings-l-model cnn
python evaluate.py machismo embeddings-l-model bi


echo "Testing ot..."

python evaluate.py ot fasttext lstm
python evaluate.py ot fasttext cnn
python evaluate.py ot fasttext bi

python evaluate.py ot glove lstm
python evaluate.py ot glove cnn
python evaluate.py ot glove bi

python evaluate.py ot wiki lstm
python evaluate.py ot wiki cnn
python evaluate.py ot wiki bi

python evaluate.py ot embeddings-l-model lstm
python evaluate.py ot embeddings-l-model cnn
python evaluate.py ot embeddings-l-model bi


echo "Testing spain..."

python evaluate.py spain fasttext lstm
python evaluate.py spain fasttext cnn
python evaluate.py spain fasttext bi

python evaluate.py spain glove lstm
python evaluate.py spain glove cnn
python evaluate.py spain glove bi

python evaluate.py spain wiki lstm
python evaluate.py spain wiki cnn
python evaluate.py spain wiki bi

python evaluate.py spain embeddings-l-model lstm
python evaluate.py spain embeddings-l-model cnn
python evaluate.py spain embeddings-l-model bi


echo "Testing varw..."

python evaluate.py varw fasttext lstm
python evaluate.py varw fasttext cnn
python evaluate.py varw fasttext bi

python evaluate.py varw glove lstm
python evaluate.py varw glove cnn
python evaluate.py varw glove bi

python evaluate.py varw wiki lstm
python evaluate.py varw wiki cnn
python evaluate.py varw wiki bi

python evaluate.py varw embeddings-l-model lstm
python evaluate.py varw embeddings-l-model cnn
python evaluate.py varw embeddings-l-model bi

echo "Testing full_corpus..."

python evaluate.py full_corpus fasttext lstm
python evaluate.py full_corpus fasttext cnn
python evaluate.py full_corpus fasttext bi

python evaluate.py full_corpus glove lstm
python evaluate.py full_corpus glove cnn
python evaluate.py full_corpus glove bi

python evaluate.py full_corpus wiki lstm
python evaluate.py full_corpus wiki cnn
python evaluate.py full_corpus wiki bi

python evaluate.py full_corpus embeddings-l-model lstm
python evaluate.py full_corpus embeddings-l-model cnn
python evaluate.py full_corpus embeddings-l-model bi

