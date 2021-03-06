﻿'''
Python program source code
for research article "Variable-Depth Convolutional Neural Network for Text Classification"

Version 1.0
(c) Copyright 2020 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The python program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The python program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''


import os
import torch
import torchtext

#python -m spacy download en_core_web_sm
Text = torchtext.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True, batch_first=True)
Label = torchtext.data.LabelField(is_target=True)

trainData, devData, testData = torchtext.datasets.SST.splits(Text, Label)
print('Dataset Size:', len(trainData), len(devData), len(testData))

Text.build_vocab(trainData.text, vectors="glove.6B.300d")
Label.build_vocab(trainData.label)
print('Text Vocabulary Size:', len(Text.vocab))
print('Label Vocabulary Size:', len(Label.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################################

class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, tensor):
        return tensor.transpose(self.dim0, self.dim1)

class VariableDepth(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.Embedding = torch.nn.Sequential(
            torch.nn.Embedding.from_pretrained(Text.vocab.vectors), #glove.6B.300d
            torch.nn.Dropout(0.2, inplace=True),
            )

        self.CNN = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1024, kernel_size=(11,300), padding=(5,0)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(11,1), padding=(5,0)),
            )

        self.AdaptConnect = torch.nn.Sequential(
            Transpose(1, 3),
            torch.nn.Linear(1024, 300),
            torch.nn.Softsign(),
            )

        self.FullyConnect = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(300, 3),
            )

    def forward(self, text):
        feature = self.Embedding(text).unsqueeze(1) #[batch_size, 1, S, E]

        while (feature.size(2) > 1):
            feature = self.CNN(feature) #[batch_size, F, S, 1]
            feature = self.AdaptConnect(feature) #[batch_size, 1, S, E]

        return self.FullyConnect(feature) #[batch_size, C]

#####################################################################################

epoch = 0
VariableDepth = VariableDepth().to(device)
Optimizer = torch.optim.Adagrad(VariableDepth.parameters())
#Optimizer = torch.optim.Adam(VariableDepth.parameters())
if os.path.exists('checkpoint_SST.pkl'):
    print("Loaded checkpoint_SST.pkl")
    checkpoint = torch.load('checkpoint_SST.pkl')
    epoch = checkpoint['epoch']
    VariableDepth.load_state_dict(checkpoint['VariableDepth.state_dict'])
    Optimizer.load_state_dict(checkpoint['Optimizer.state_dict'])
print(VariableDepth)
print(Optimizer)

for parameter in VariableDepth.parameters():
    print(parameter.shape)

trainItr, devItr, testItr = torchtext.data.BucketIterator.splits((trainData, devData, testData), batch_size=100, sort=False, device=device)
loss_func = torch.nn.CrossEntropyLoss()

while (epoch < 200):
    torch.cuda.empty_cache()
    epoch += 1
 
    sumLoss = 0.0
    VariableDepth.train()
    with torch.enable_grad():
        for step, train in enumerate(trainItr):
            Optimizer.zero_grad()

            label = VariableDepth(train.text)
            loss = loss_func(label, train.label)
            #print(f'{step:05} | Train Loss: {loss.data:.4f}')
            sumLoss += loss.data

            loss.backward()
            Optimizer.step()
    sumLoss /= len(trainItr)
    print(f'Epoch: {epoch:02} | Average Loss: {sumLoss:.4f}')

    #continue
    accuracy = 0.0
    VariableDepth.eval()
    with torch.no_grad():
        for step, test in enumerate(testItr):
            label = VariableDepth(test.text)
            accuracy += (label.max(-1)[1] == test.label).sum()
    accuracy /= len(testData)
    print(f'Epoch: {epoch:02} | Test Accuracy: {accuracy:.4f}')

    continue
    torch.save(
       {
           'epoch': epoch,
           'VariableDepth.state_dict': VariableDepth.state_dict(),
           'Optimizer.state_dict': Optimizer.state_dict()
           },
       'checkpoint_SST-%03d-(%.04f,%.04f).pkl' %(epoch, sumLoss, accuracy)
       )
