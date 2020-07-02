from parlai.core.agents import create_agent
from parlai.core.opt import Opt
from parlai.core.worlds import validate

from ..utils import registry, language
from ..google import translate


@registry.skill()
async def parlai(bot, userstate, initial=None):
    opt = Opt.load('opt')
    agent = create_agent(opt, requireModelExists=True)
    user_text = initial or await bot.listen_text(timeout=10)
    while True:
        if language.get() != 'en':
            user_text = await translate(language.get(), 'en', user_text)
        msg = dict(type='message', text=user_text, episode_done=False)
        agent.observe(validate(msg))
        response = agent.act()
        bot_text = response['text'].replace(" ' ", "'").replace(' ,', ',')
        if language.get() != 'en':
            bot_text = await translate('en', language.get(), bot_text)
        user_text = await bot.ask(bot_text, timeout=10)
