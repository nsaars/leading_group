from aiogram import types
from aiogram.enums import ParseMode
from aiogram.fsm.context import FSMContext
from aiogram.types import InputMediaPhoto, FSInputFile

from utils.ai_assistant.ai_chain import AiChain


async def ai_conversation_handler(message: types.Message, state: FSMContext):
    if message.content_type != "text":
        await message.answer("Меня настроили отвечать только на текстовые сообщения :)")
        return

    state_data = await state.get_data()

    history: list = state_data.get('history') or []
    response: dict = await AiChain.get_proper_response(message.text, history)

    await state.update_data({'history': history + [('user', message.text), ('assistant', response.get('text'))]})
    #update_state(state_data.get('db_state_id'),
    #             {'title': 'ai_conversation', 'data': await state.get_data()})  # todo: custom fsm context
    #create_message(state_data.get('db_user_id'), 'user', message.text)
    #create_message(state_data.get('db_user_id'), 'assistant', response.get('text'), response.get('type'))

    media_group = []
    images = response.get('images')
    if images:
        for image in images:
            media_group.append(InputMediaPhoto(media=FSInputFile(image), caption=response.get('text')))

        await message.answer_media_group(media=media_group, parse_mode=ParseMode.MARKDOWN)
    else:

        await message.answer(response.get('text'), parse_mode=ParseMode.MARKDOWN)

    # kwargs = response.get('schedule_consultation_kwargs')
    # if kwargs and response.get('success'):
    # await send_consultation_request(message.bot, state_data.get('db_user_id'), history, **kwargs)
