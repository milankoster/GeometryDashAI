# Snake Q Learning
This project uses two project reinforcement learning approached to train an agent to play Geometry Dash. The aim is to gain experience in applying reinforcement learning to a pre-built environment and explore the feasibility of reinforcement learning in Geometry Dash. 

## Reinforcement Learning
The first approach to the problem was to use basic reinforcement learning. While this approach was not successful, a number of configurations were attempted. These variations included, but were not limited to:

- Various model configurations.
- Various reward functions.
- Use of a target model.
- Episodes: 500, 1000, 2000.
- Image size: 160, 320, 640.
- Epsilon decay rate: 0.95, 0.99, 0.995.
- Epsilon minimum: 0.001, 0.01, 0.10.
- Training queue: 1000, 2500.

The most impactful of these were changes to the reward function. Punishing jumping to discourage random jumps results in the model never jumping. However, when solely awarding level progress the model would fail to consistently pick up patterns.  

In some cases, the model would learn to perform quite well, but worsen over time. The main hypothesis is that this is due to the low framerate. While the screen can be recorded at 30 to 60 frames per second, model predictions slow this down significantly. A potential solution would be to run predictions in parallel. 

## Imitation Learning
The second approach involved using imitation learning. The model was trained using a dataset of 6 successful runs of the first level, allowing it to imitate human behavior. 

The use of imitation learning significantly improved the model's performance, enabling it to understand when to jump and achieve greater progress in the game levels. 

However, while the model was able to get much further, it was unable to complete the level. One of the reasons for this is once again the framerate. While the model can accurate predict during which frame to jump, it will still die if it never gets to see the correct frame.

The model is also able to play small portions of levels it has never seen, assuming the environment looks similar. This confirms it is able to abstract some basic principles. Of course, when the model runs into obstacles or mechanics it has never encountered, it does not perform well.

In the future, it may be interesting to see whether the model from this approach can be developed further using other reinforcement learning techniques.