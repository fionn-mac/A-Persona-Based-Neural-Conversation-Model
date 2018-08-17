import random

import torch

class TrainNetwork(object):
    def __init__(self, encoder, decoder, index2word, max_length, num_layers=1,
                 batch_size=1, teacher_forcing_ratio=0):
        self.encoder = encoder
        self.decoder = decoder
        self.index2word = index2word
        self.num_layers = num_layers
        self.SOS_token = 1
        self.EOS_token = 2
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def train(self, input_variables, target_variables, people, lengths, encoder_optimizer,
              decoder_optimizer, criterion):

        target_length = target_variables.size()[0]

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0

        encoder_hidden = self.encoder.initHidden()

        encoder_outputs, encoder_hidden = self.encoder(input_variables, lengths, encoder_hidden)

        decoder_inputs = torch.LongTensor([[self.SOS_token]*self.batch_size])
        decoder_inputs = decoder_inputs.cuda() if self.use_cuda else decoder_inputs

        # Decoder Unidirectional while Encoder Bidirectional
        decoder_hidden = encoder_hidden[:self.num_layers].view(self.num_layers, self.batch_size, -1)

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_outputs, decoder_hidden, _ = self.decoder(decoder_inputs, people, decoder_hidden,
                                                                  encoder_outputs, lengths)
                loss += criterion(decoder_outputs, target_variable[di])
                decoder_inputs = target_variables[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_outputs, decoder_hidden, _ = self.decoder(decoder_inputs, people, decoder_hidden,
                                                                  encoder_outputs, lengths)
                topv, topi = decoder_outputs.data.topk(1)
                decoder_inputs = topi.permute(1, 0)
                loss += criterion(decoder_outputs, target_variables[di])

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def evaluate(self, input_variable, people):

        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.initHidden(1)

        encoder_outputs, encoder_hidden = self.encoder(input_variable, [input_length], encoder_hidden)

        decoder_input = torch.LongTensor([[self.SOS_token]])  # SOS
        if self.use_cuda: decoder_input = decoder_input.cuda()

        # Decoder Unidirectional while Encoder Bidirectional
        decoder_hidden = encoder_hidden[:self.num_layers].view(self.num_layers, 1, -1)

        decoded_words = []
        decoder_attentions = torch.zeros(self.max_length, input_length)

        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, people, decoder_hidden,
                                                                             encoder_outputs, [input_length])
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == self.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.index2word[int(ni)])

            decoder_input = torch.LongTensor([[ni]])
            if self.use_cuda: decoder_input = decoder_input.cuda()

        return decoded_words, decoder_attentions[:di + 1]
