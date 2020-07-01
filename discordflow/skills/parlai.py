from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.worlds import validate

from ..utils import registry


@registry.skill()
async def parlai(bot, userstate):
    opt = Opt.load('opt')
    agent = create_agent(opt, requireModelExists=True)
    user_text = await bot.listen_text(timeout=10)
    while True:
        msg = dict(type='message', text=user_text, episode_done=False)
        agent.observe(validate(msg))
        response = agent.act()
        bot_text = response['text'].replace(" ' ", "'").replace(' ,', ',')
        user_text = await bot.ask(bot_text, timeout=10)
