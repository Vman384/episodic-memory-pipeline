`Identity`: I am an engineer planning on designing a benchmarking pipeline to asssess VLM's episodic memory capacilities. You are meant to design and implement the code alongside me.

RULES:

1. Do not over engineer anything, only implement what was told.
2. Follow the KISS principle, Keep it simple, stupid.
3. Update the documentations in CODEBASE_DOCUMENTATION.md, README.md when code is written that makes the documentation outdated.

`Task Context`: Develop an episodic memory benchmark. This benchmark aims to target 4 distinct categories, namely:
     - **Sparse event localisation**
          Temporal perception, existence, noteworthy object and moments queries, deceptive questioning
          We are benchmarking models' ability to verify the existence of rare, outlier events without hallucinating things.
          Ensure the model is not guessing based on the context of the video, but instead actually remembers what happened.
     - **temporal order**
          Benchmarking whether model actually remembers the order of events, or just guessing based on previous training
     - **Spatial Reasoning**
          Aims to assess models ability of physically grounded reasoning like estimating distances. 
          Where in places like forest, rural areas or tunnels, reference points are scarce, or not as seen in datasets trained.
     - **Counting**
          In high SNR, objects are everywhere, thus models fail because they lose count. 
          But in our case, the challenge is instead episodic memory, where the model cannot rely on a single frame, 
          but episodic memory to remember how many we saw.

     - **Note** 

          The inputs are camera vision based, egocentric of a vehicle driving around 3 environnments:
               - Forests
               - Rural roads
               - Tunnels
          Each videos will range around 6 minutes to 1 hour. Thus this assess the model's episodic memory recall capabilities
          in an extremely long videos, with very minimal events going on.




