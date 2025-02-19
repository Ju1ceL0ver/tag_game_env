from tag import Player,EnvTag,ActorCriticTag,Point
from torchvision.transforms.functional import to_pil_image
model=ActorCriticTag()
player=Player(1,model,100,100,-1)
point=Point(50,50)
env=EnvTag(point,player)

to_pil_image(env.draw().squeeze(0)).show()