from modules.Attention import *
from modules.Drop import *


class HANEncoder(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(HANEncoder, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False

        self.role_embed = nn.Embedding(vocab.role_size, config.role_dims, padding_idx=0)
        role_init = np.zeros((vocab.role_size, config.role_dims), dtype=np.float32)
        self.role_embed.weight.data.copy_(torch.from_numpy(role_init))

        self.sent_lstm = MyLSTM(
            input_size=config.word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.turn_lstm = MyLSTM(
            input_size=config.lstm_hiddens * 2,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.sent_att = Attention(config.lstm_hiddens * 2)
        self.turn_att = Attention(config.lstm_hiddens * 2)

    def forward(self, words, extwords, roles, sent_masks, turn_masks):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed

        b, turn_size, length, h = x_embed.size()

        x_embed = x_embed.view(-1, length, h)
        sent_masks = sent_masks.view(-1, length)

        if self.training:
            x_embed = drop_input_independent(x_embed, self.config.dropout_emb)

        sent_lstm_hiddens, _ = self.sent_lstm(x_embed, sent_masks, None)
        sent_lstm_hiddens = sent_lstm_hiddens.transpose(1, 0)

        #if self.training:
            #sent_lstm_hiddens = drop_sequence_sharedmask(sent_lstm_hiddens, self.config.dropout_mlp)

        sent_represents = self.sent_att(sent_lstm_hiddens, sent_masks)

        sent_represents = sent_represents.view(b, turn_size, -1)

        #role_emb = self.role_embed(roles)

        #sent_represents = torch.cat([sent_represents, role_emb], -1)

        turn_lstm_hiddens, _ = self.turn_lstm(sent_represents, turn_masks, None)
        #if self.training:
            #turn_lstm_hiddens = drop_sequence_sharedmask(turn_lstm_hiddens, self.config.dropout_mlp)
        turn_lstm_hiddens = turn_lstm_hiddens.transpose(1, 0)

        diaglog_represents = self.turn_att(turn_lstm_hiddens, turn_masks)  # b hidden

        return diaglog_represents

