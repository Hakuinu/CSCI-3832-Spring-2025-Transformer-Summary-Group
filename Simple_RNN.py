import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch import optim

# Implemented by using class notes, pytorch.org tutorials, and modifying them but also using DeepSeek for ideas here and there

class LSTM(nn.Module):

	#Freeze embeddings is so the embedding layer isnt updated during training.
	def __init__(self, pretrained_embeddings, hidden_dim, output_dim, freeze_embeddings=True):
		super().__init__()

		# Get embedding dimensions from pretrained embeddings
		vocab_size, embedding_dim = pretrained_embeddings.shape
		
		# Create embedding layer with pretrained weights
		self.embedding = nn.Embedding.from_pretrained(
			torch.FloatTensor(pretrained_embeddings),
			freeze=freeze_embeddings
		)
		
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
		self.classification_hidden_layer = nn.Linear(hidden_dim, hidden_dim)
		self.output_layer = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()

	def forward(self, inputs, input_lengths):

		embedding = self.embedding(inputs)

		packed_input = pack_padded_sequence(embedding, input_lengths, batch_first=True, enforce_sorted=False)

		output, (hn, _) = self.lstm(packed_input) # What is the difference between output and hn?

		output, output_lengths = pad_packed_sequence(output, batch_first=True)

		# print(output.shape) # Shape: batch_size x sequence_length x hidden_dim

		h1 = self.classification_hidden_layer(output)

		h1 = self.relu(h1)

		final_output = self.output_layer(h1)

		return final_output
	
def main():
	pretrained_embeddings = torch.load('review_embeddings.pt', weights_only=True) # Weights only set to true due to a weird warning
	model = LSTM(
		pretrained_embeddings=pretrained_embeddings,
		#both these parameters can be updated
		hidden_dim=256, 
		output_dim=2,  
		freeze_embeddings=True
	)

	


if __name__ == "__main__":
	main()
