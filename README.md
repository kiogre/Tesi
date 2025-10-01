# Thesis
This is my work about the thesis, it is a neural network that should give as output the right order to make the operations in a JSP (Job Shop Problem)
It is inspired by the work of the paper that is on this github (Lion17) where they did a neural network with the encoder made by dense layers and then a Multiattention layer. The decoder is a LSTM and a pointer layer (the decoder is a pointer network) that points to an element of the input, there is a mask to prevent making an impossible choice.
