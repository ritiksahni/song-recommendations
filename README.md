I was curious about how I'd make a song recommendations engine work. This is a swing at it.

- Uses the audio data (tempo, timbre, rhythm, energy)
- Creates a vector store.
- When a song is input, it performs similarity search and returns the closest matches.
