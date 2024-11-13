import re
import inflect
import spacy
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
)


def get_prompt(query, model, model_name, args):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


words_to_drop = [
    "image",
    "photo",
    "picture",
    "thumbnail",
    "logo",
    "symbol",
    "clipart",
    "portrait",
    "painting",
    "illustration",
    "icon",
    "profile",
    "feature",
    "pair",
    "group",
    "front",
    "back",
    "left",
    "right",
    "foreground",
    "background",
    "context",
    "second",
    "minute",
    "hour",
    "morning",
    "afternoon",
    "evening",
    "night",
    "midnight",
    "noon",
    "weekday",
    "weekend",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def extract_items(text):

    # ToLower
    text = text.strip().lower()

    if text.endswith('.'):
        text = text[:-1]
    phrases = text.split(',')
    phrases = [phrase.strip() for phrase in phrases]

    # Remove Duplicates
    phrases = list(set(phrases))

    # Post-Processing
    singular_noun_transform = inflect.engine().singular_noun
    singular_phrases = []
    for phrase in phrases:
        # if not isinstance(phrase, str):
        #     phrase = str(phrase)
        # Drop Words
        if phrase in words_to_drop:
            continue

        # Singular_noun
        if phrase[-2:] in ["ss", "us", "is", "ns"] or phrase[-3:] in ["ies", "oes"]:
            pass
        elif phrase.endswith("s"):

            try:
                phrase = singular_noun_transform(phrase) or phrase
            except TypeError as e:
                pass

        if phrase == "men":
            phrase = "man"
        elif phrase == "women":
            phrase = "woman"

        singular_phrases.append(phrase)

    singular_phrases = [phrase for phrase in singular_phrases if phrase != ""]

    return singular_phrases


def get_llava_output(image, image_processor, input_prompt, model, tokenizer, args):
    image_sizes = [x.size for x in image]

    images_tensor = process_images(image, image_processor, model.config).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(input_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # # set stopping_criteria
    # keywords = ['.']
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            # stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def generate_coa(image, image_processor, model, model_name, tokenizer, args):
    question1 = ("What are the main objects in this image? Please provide a one-sentence caption for the image that "
                 "includes as many objects as possible. Do not attempt to describe the environment or atmosphere "
                 "depicted in the image.")
    # Caption Action
    prompt1 = get_prompt(question1, model, model_name, args)
    outputs1 = get_llava_output(image, image_processor, prompt1, model, tokenizer, args)
    vocabularies1 = extract_items(outputs1)

    # Self-Correct Action
    objects = []
    object_features = []
    for object in vocabularies1:
        question = f"Please determine whether the image contains a '{object}'. Just reply with 'Yes' or 'No'."
        prompt = get_prompt(question, model, model_name, args)
        outputs = get_llava_output(image, image_processor, prompt, model, tokenizer, args)
        if outputs == 'Yes':
            objects.append(object)

    # Appearance Action
    for object in objects:
        next_question = (f"For the object '{object}' shown in this image, please provide a brief description of its "
                         f"main appearance and features.")
        next_prompt = get_prompt(next_question, model, model_name, args)
        next_outputs = get_llava_output(image, image_processor, next_prompt, model, tokenizer, args)
        object_features.append(next_outputs)

    objects_text = ', '.join(objects)
    object_features_text = ', '.join(object_features)

    # Relationship Action
    question3 = (f"Given a sequence of objects: <{objects_text}>. Here is a list of object description text: "
                 f"<{object_features_text}>. Please answer the following question: \n "
                 f"What relationships exist between these objects? Please answer with a brief description.")
    prompt3 = get_prompt(question3, model, model_name, args)
    outputs3 = get_llava_output(image, image_processor, prompt3, model, tokenizer, args)


    # Final Action
    question5 = (f"Given a sequence of objects: <{objects_text}>. Here is a list of object description text: "
                 f"<{object_features_text}>. Here is the relationships exist between these objects: <{outputs3}>. "
                 f"Based on these information, please answer the following question: \n "
                 f"What are the main physical objects are in this image? Just output the object names without details.")
    prompt5 = get_prompt(question5, model, model_name, args)
    outputs5 = get_llava_output(image, image_processor, prompt5, model, tokenizer, args)
    vocabularies5 = extract_items(outputs5)

    return vocabularies5, outputs5