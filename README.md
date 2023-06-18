# Snake Q Learning
This project explores two project reinforcement learning approaches to train an agent to play Geometry Dash. The aim is to gain experience in applying reinforcement learning to a pre-built environment and assess the feasibility of reinforcement learning in Geometry Dash. 

## Reinforcement Learning
In the first approach we utilised basic reinforcement learning. Although this was not successful, several configurations were attempted including various model architectures, reward functions, episode lengths, image sizes and epsilon decay values. We also played around with a target model to stabilise training.

Changes to the reward function were the most impactful. Punishing jumping to discourage random jumps resulted in the model never jumping. However, solely rewarding progress in the level caused the model to struggle in consistently identifying patterns.

In some cases, the model would initially learn to perform quite well but worsen over time. The main hypothesis is that this is due to the low framerate. While the screen can be recorded at 30 to 60 frames per second, model predictions slow this process down significantly. A potential solution would be to run predictions in parallel. 

## Imitation Learning
The second approach involved using imitation learning. The model was trained using a dataset consisting of 6 successful runs of the first level, allowing it to imitate human behavior.

Imitation learning significantly improved the model's performance, enabling it to understand when to jump and achieve greater progress in the game levels. 

Although the model is able to get much further than before, it is unable to complete the level. One of the reasons for this limitation is once again the framerate issue. While the model can accurately predict the appropriate frame to jump, it still fails when it is not shown the correct frame.

The model is also able to play small portions of levels it has never seen, assuming the environment looks similar. This confirms it is able to abstract some basic principles. 

In future work, it would be interesting to explore the further development of this model using other reinforcement learning techniques.
