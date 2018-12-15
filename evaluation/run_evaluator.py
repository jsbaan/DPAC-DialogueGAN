from seq2seq.TopKDecoder import TopKDecoder
from helpers import *
from generator import Generator
from Evaluator import Evaluator

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')  #'
else:
    DEVICE = torch.device('cpu')  #'cuda:0'

VOCAB_SIZE = 8000
MIN_SEQ_LEN = 5
MAX_SEQ_LEN = 20
BATCH_SIZE = 256
GEN_EMBEDDING_DIM = 256
GEN_HIDDEN_DIM = 256

if __name__ == '__main__':
    evaluator = Evaluator(vocab_size=VOCAB_SIZE, min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE)

    result = {}
    for i in range(0, 32):
        gen = Generator(evaluator.sos_id, evaluator.eou_id, VOCAB_SIZE, GEN_HIDDEN_DIM, GEN_EMBEDDING_DIM, MAX_SEQ_LEN, teacher_forcing_ratio=0)

        model_path = '../generator_checkpoint' + str(i) + '.pth.tar'
        data = torch.load(model_path, map_location='cpu')
        gen.load_state_dict(data['state_dict'])
        gen.decoder = TopKDecoder(gen.decoder, 5)
        gen.to(DEVICE)

        print('Evaluating ' + model_path)
        result[i] = evaluator.evaluate_embeddings(gen)

    print('Result')
    print(result)