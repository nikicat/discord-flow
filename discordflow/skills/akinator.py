import json
import logging
import re

import aiohttp

from ..utils import registry
from ..google import set_contexts

logger = logging.getLogger(__name__)


class Unauthorized(Exception):
    pass


class ResponseError(Exception):
    pass


def parse_response(text):
    json_ = re.sub(r'jquery\((.*)\)', r'\1', text)
    if json_ == '{["completion" => "KO - UNAUTHORIZED"]}':
        raise Unauthorized
    resp = json.loads(json_)
    logger.debug(f"Akinator response: {resp}")
    if resp['completion'] != 'OK':
        raise ResponseError
    return resp['parameters']


def get_url_api_ws():
    return 'https://srv12.akinator.com:9398/ws'


async def answer_api(sess: aiohttp.ClientSession, step: int, answer: int, session: int, signature: int, frontaddr: str):
    resp = await sess.get(
        'https://ru.akinator.com/answer_api',
        params=dict(
            callback='jquery',
            urlApiWs=get_url_api_ws(),
            session=session,
            signature=signature,
            step=step,
            answer=answer,
            frontaddr=frontaddr,
        ),
        headers={'x-requested-with': 'XMLHttpRequest'},
    )
    return parse_response(await resp.text())


async def new_session(sess, uid_ext_session, frontaddr):
    resp = await sess.get(
        'https://ru.akinator.com/new_session',
        params=dict(
            callback='jquery',
            urlApiWs=get_url_api_ws(),
            player='website-desktop',
            partner=1,
            uid_ext_session=uid_ext_session,
            frontaddr=frontaddr,
            childMod='',
            constraint="ETAT<>'AV'",
            soft_constraint='',
            question_filter='',
        ),
        headers={'x-requested-with': 'XMLHttpRequest'},
    )
    return parse_response(await resp.text())


async def fetch_main(sess):
    resp = await sess.get('https://ru.akinator.com/game')
    body = await resp.text()
    uid_ext_session = re.search(r"var uid_ext_session = '([a-z0-9\-]+)';", body).group(1)
    frontaddr = re.search(r"var frontaddr = '([a-zA-Z0-9+/=]+)';", body).group(1)
    return uid_ext_session, frontaddr


async def call_list(sess, session, signature, step):
    return await call_api(
        sess, session, signature, step, 'list', size=2, max_pic_width=246, max_pic_height=294, pref_photos='VO-OK',
        duel_allowed=1, mode_question=0,
    )


async def call_exclusion(sess, session, signature, step):
    return await call_api(sess, session, signature, step, 'exclusion', question_filter='', forward_answer=1)


async def call_api(sess, session, signature, step, api, **params):
    resp = await sess.get(
        f'{get_url_api_ws()}/{api}',
        params=dict(
            callback='jquery',
            session=session,
            signature=signature,
            step=step,
            **params,
        ),
    )
    return parse_response(await resp.text())


async def call_choice(sess, session, signature, step, element):
    return await call_api(sess, session, signature, step, 'choice', element=element, duel_allowed=1)


@registry.skill()
async def akinator(bot, userstate):
    async with aiohttp.ClientSession() as sess:
        uid_ext_session, frontaddr = await fetch_main(sess)
        info = await new_session(sess, uid_ext_session, frontaddr)
        ident = info['identification']
        step = info['step_information']
        session = ident['session']
        signature = ident['signature']
        to_speak = step['question']
        while True:
            with set_contexts('akinator'):
                intent = await bot.ask_and_detect_intent(to_speak)
                if intent.action == 'akinator.answer':
                    step = await answer_api(
                        sess,
                        step=step['step'],
                        answer=intent.parameters['answer'],
                        session=session,
                        signature=signature,
                        frontaddr=frontaddr,
                    )
                    logger.info(f"Step {step['step']} = {step['progression']}% {step['question']}")
                    if float(step['progression']) > 97:
                        choices = await call_list(sess, session, signature, step['step'])
                        choice = choices['elements'][0]['element']
                        agreed = await bot.ask_yes_no(f"Это {choice['name']}?")
                        if agreed:
                            resp = await call_choice(sess, session, signature, step['step'], element=choice['id'])
                            times_selected = resp['element_informations']['times_selected']
                            await bot.say(f"Я как всегда молодец! Персонаж уже был отыгран {times_selected} раз")
                            break
                        else:
                            resp = await call_exclusion(sess, session, signature, step['step'])
                            agreed = await bot.ask_yes_no("Продолжаем?")
                            if agreed:
                                to_speak = step['question']
                            else:
                                await bot.say("Спасибо за игру!")
                                break
                    else:
                        to_speak = step['question']
                else:
                    answers = '; '.join(a['answer'] for a in step['answers'])
                    to_speak = f"Отвечай: {answers}"
