"""
generation.py
--------------
Vastausten generointi LumiOpen/Viking-13B -mallilla.
Painotus: tiukka RAG – vastaa vain lähdemateriaalin perusteella.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_error()


def generate_answer(question: str, context: list[str]):
    """Generoi tiukan, suomenkielisen vastauksen Viking-13B-mallilla."""
    print("\n⚙️ Generoidaan vastaus mallilla LumiOpen/Viking-13B...")

    model_name = "LumiOpen/Viking-13B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    if not context:
        print("⚠️ Konteksti on tyhjä — ei kappaleita, joista vastata.")
        return "En pysty vastaamaan, koska lähteitä ei löytynyt."

    # 🎯 Tiukka, ohjaava suomenkielinen system prompt
    system_prompt = (
        "Toimit suomenkielisenä tekoälyavustajana, joka vastaa vain annettujen lähteiden perusteella.\n"
        "Tehtäväsi on kertoa, miten verkkolähde merkitään lähdeluetteloon suomalaisessa opinnäytetyössä.\n"
        "Käytä vain alla annettua kontekstia – älä keksi omaa sisältöä.\n"
        "Jos kontekstissa ei ole ohjetta verkkolähteen merkitsemiseen, vastaa: "
        "'En löydä varmaa ohjetta annetuista lähteistä.'\n\n"
        "Vastauksesi tulee olla lyhyt (2–4 lausetta) ja sisältää konkreettinen esimerkki muodossa:\n"
        "Tekijä. Vuosi. Otsikko. Verkkosivusto. Saatavilla: URL. Viitattu pp.kk.vvvv.\n"
        "Älä lisää mitään muuta tekstiä.\n"
    )

    # 🧩 Rakennetaan konteksti – vain olennaisimmat kappaleet
    ctx_text = ""
    for i, p in enumerate(context[:5]):
        if len(tokenizer.encode(ctx_text + p)) > 1500:
            break
        ctx_text += f"[Kappale {i+1}]\n{p}\n\n"

    # 🔤 Lopullinen prompt
    prompt = (
        f"{system_prompt}"
        f"Kysymys: {question}\n\n"
        f"Alla on lähdeaineistosta poimitut kappaleet:\n"
        f"{ctx_text}\n\n"
        "Kirjoita vastaus vain näiden kappaleiden pohjalta.\n\nVastaus:"
    )

    # 🧮 Tokenointi ja generointi
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.15,
        )

    answer = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # 🔍 Validointi — tarkista, ettei malli harhaile
    key_terms = ["lähdeluettelo", "verkkolähde", "viitattu", "Saatavilla"]
    if not any(k in answer.lower() for k in key_terms):
        print("⚠️ Mallin vastaus ei sisältänyt aiheeseen liittyviä avainsanoja — yritetään uudelleen vähemmällä lämmöllä.")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.2,
                top_p=0.7,
                do_sample=True,
                repetition_penalty=1.2,
            )
        answer = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # 🧹 Puhdistetaan lopputulos
    for unwanted in ["\n\n", "\n", "###", "Vastaus:", "Lähteet:", "Kysymys:"]:
        if answer.startswith(unwanted):
            answer = answer.replace(unwanted, "").strip()

    print("\n📝 Generoitu vastaus valmiina.\n")
    return answer
