''' 
arguements
- config path
- device (default: cuda:0)
'''

#  load config

# in config there will be an expert policy path
# load the expert policy

# in config there will be a q function path
# load the q function

# in config there will be a pi search config path
# load the pi search config

# load the pi search model with the config

# generate action chunks with noised state

# compute compute the q values for the action chunks

# pass those back into pi search to get the new action chunks
