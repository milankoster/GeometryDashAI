# Snake Q Learning
This project explores two project reinforcement learning approaches to train an agent to play Geometry Dash. The aim is to gain experience in applying reinforcement learning to a pre-built environment and assess the feasibility of reinforcement learning in Geometry Dash. 

## Reinforcement Learning
In the first approach I applied basic reinforcement learning. Although this approach was unsuccessful, I made numerous attempts to fine-tune the system by experimenting with different configurations. These included variations in model architectures, reward functions, episode lengths, image sizes and epsilon decay values. Additionally, I explored the use of a target model to increase training stability. 

Among the modifications, altering the reward function had the most impact. To discourage random jumps we penalized this behavior, but this led to the model never jumping at all. However, when we solely rewarded progress in the level, the model struggled to consistently identify patterns.

![image](https://github.com/milankoster/GeometryDashAI/assets/58393068/a3e5021b-3b84-4fe6-bc85-2ff833d86c09)

In some instances, the model initially performed quite well but worsened over time. My primary hypothesis is that this is due to the low framerate. While the screen recorder was able to capture 30 to 60 frames per second, the model’s predictions significantly slowed this process down. A potential solution to mitigate this issue would be to run predictions in parallel.

## Imitation Learning
The second approach involved using imitation learning. The model was trained using a dataset consisting of 6 successful runs of the first level, allowing it to imitate human behaviour. 

The imitation learning model was built as a custom CNN model, with an input of 320 by 320 pixels, 3 convolutional layers, and 2 output neurons. The full details can be found in the [Imitation Learning](https://github.com/milankoster/GeometryDashAI/blob/master/Imitation%20Learning.ipynb) notebook. The model was tested on both a validation and testing set, with an accuracy of approximately 92%. 

![image](https://github.com/milankoster/GeometryDashAI/assets/58393068/040784dc-4e62-40ba-8c0f-49e67cf0a9ad)

It is possible to introduce a bias, by changing the prediction threshold as described in chapter 6.4. However, when evaluating the model on the game, I found that this was not necessary. Imitation learning significantly improved the model's performance, enabling it to understand when to jump and achieve greater progress in the game levels.

![image](https://github.com/milankoster/GeometryDashAI/assets/58393068/5d4c9418-bf7e-4117-ae6d-8f70321f4377)

Although the model is able to get much further, it is not able to complete any levels. One of the reasons for this limitation is once again the framerate issue. While the model can accurately predict the appropriate frame to jump, it still fails when it is not shown the correct frame. The model is also able to play small portions of levels it has never seen, assuming the environment looks similar. This confirms it is able to abstract some basic principles.

In future work, it would be interesting to explore the further development of this model using other reinforcement learning techniques.
