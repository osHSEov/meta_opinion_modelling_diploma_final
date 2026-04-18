import json
import re
import argparse
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import ollama
from tqdm import tqdm
import random

@dataclass
class SyntheticSample:
    id: str
    topic: str
    text: str
    agents: List[str]
    propositions: List[str]
    formulas: List[str]
    depth: int
    metadata: Dict
    
class MetaOpinionDatasetGenerator:
    def __init__(
        self,
        model_name: str = "gpt-oss:120b",
        temperature: float = 0.75,
        seed: Optional[int] = 42,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        
    def build_system_prompt(self) -> str:
        return "Ты — эксперт по эпистемической логике и генерации синтетических данных для бенчмарка извлечения мета-мнений."
    def generate_diverse_topics(self, num_topics: int = 50) -> List[str]:
        """Генерирует свежий список разнообразных тем с помощью LLM"""
        print(f"Генерирую {num_topics} разнообразных тем для мета-мнений...")

        prompt = f"""Сгенерируй ровно {num_topics} коротких, реалистичных и спорных тем, которые идеально подходят для обсуждения мета-мнений (мнений о мнениях других людей).
Темы должны быть из разных сфер жизни. Распредели примерно поровну по категориям:
- Здоровье и медицина
- Экология и климат
- Технологии и ИИ
- Политика и общество
- Экономика и криптовалюта
- Социальные нормы и психология
- Этика и философия
- Работа и образование

Каждая тема — 3–8 слов, в стиле реального обсуждения в Reddit/Twitter.
Ответь **только** JSON-массивом строк, например:
["Вакцина вызывает бесплодие?", "ИИ скоро заберёт все рабочие места", ...]

Не добавляй нумерацию, комментарии или лишний текст."""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.8,
                    "seed": self.seed,
                    "num_ctx": 131072,
                },
            )
            content = response["message"]["content"]

            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                topics = json.loads(match.group(0))
                if isinstance(topics, list) and len(topics) > 0:
                    topics = [t.strip() for t in topics if t.strip()][:num_topics]
                    print(f"Сгенерировано {len(topics)} тем")
                    return topics

            print("Не удалось распарсить темы, использую fallback")
        except Exception as e:
            print(f"Ошибка генерации тем: {e}")

        return [
            "Вакцина от COVID вызывает бесплодие",
            "ИИ заберёт все рабочие места к 2030 году",
            "Климатический кризис — это естественный цикл",
        ][:num_topics]
    
    def _build_user_prompt(self, topic: str, max_depth: int, num_agents: int, min_props: int, max_props: int) -> str:
        return f"""Тема: {topic}
Участвует ровно {num_agents} агентов.
Максимальная глубина B-операторов: {max_depth}.
Пропозиций: {min_props}–{max_props}.

Сгенерируй короткий диалог/цепочку комментариев и выведи **ТОЛЬКО** JSON:

{{
  "text": "...",
  "agents": [...],
  "propositions": [...],
  "formulas": ["B_Alice safe", "B_Bob B_Alice dangerous", ...],
  "depth": N
}}

"""

    def _parse_response(self, content: str) -> Optional[Dict]:
        content = re.sub(r'```json\s*|\s*```', '', content, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

        if not {"text", "agents", "propositions", "formulas", "depth"}.issubset(data.keys()):
            return None
        if not isinstance(data["depth"], int) or data["depth"] < 1:
            return None
        return data

    def generate_sample(self, topic: str, max_depth: int = 4, num_agents: int = 3, min_props: int = 2, max_props: int = 5) -> Optional[SyntheticSample]:
        prompt = self._build_user_prompt(topic, max_depth, num_agents, min_props, max_props)
        for attempt in range(self.max_retries):
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._build_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": self.temperature, "seed": self.seed + attempt, "num_ctx": 131072},
                )
                parsed = self._parse_response(response["message"]["content"])
                if parsed and parsed["depth"] <= max_depth:
                    sample_id = hashlib.md5(parsed["text"].encode()).hexdigest()[:12]
                    return SyntheticSample(
                        id=sample_id,
                        topic=topic,
                        text=parsed["text"].strip(),
                        agents=parsed["agents"],
                        propositions=parsed["propositions"],
                        formulas=parsed["formulas"],
                        depth=parsed["depth"],
                        metadata={"model": self.model_name, "temperature": self.temperature, "seed": self.seed},
                    )
            except Exception:
                continue
        return None

    def generate_dataset(self, topics: List[str], samples_per_topic: int = 10, output_file: str = "meta_opinions_dataset.jsonl", max_depth: int = 4, num_agents: int = 3, append: bool = False):
        output_path = Path(output_file)
        mode = "a" if append and output_path.exists() else "w"
        all_samples = []
        total = len(topics) * samples_per_topic

        with open(output_path, mode, encoding="utf-8") as f:
            with tqdm(total=total, desc="Генерация датасета") as pbar:
                for topic in topics:
                    for _ in range(samples_per_topic):
                        sample = self.generate_sample(topic, max_depth, num_agents)
                        if sample:
                            all_samples.append(sample)
                            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                            f.flush()
                        pbar.update(1)

        print(f"\nСгенерировано {len(all_samples)} примеров → {output_file}")
        return all_samples
    

def main():
    parser = argparse.ArgumentParser(description="Генератор мета-мнений v2.1 + авто-темы")
    parser.add_argument("--model", type=str, default="gpt-oss:120b")
    parser.add_argument("--topics", type=str, nargs="+", help="Список тем вручную")
    parser.add_argument("--auto_topics", type=int, default=0, help="Сгенерировать N автоматических тем (рекомендуется 30–100)")
    parser.add_argument("--samples", type=int, default=10, help="Примеров на тему")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--output", type=str, default="meta_opinions_dataset.jsonl")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true")

    args = parser.parse_args()

    generator = MetaOpinionDatasetGenerator(model_name=args.model, temperature=args.temperature, seed=args.seed)

    if args.auto_topics > 0:
        topics = generator.generate_diverse_topics(args.auto_topics)
        Path("generated_topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Темы сохранены в generated_topics.json")
    else:
        topics = args.topics or ["vaccine efficacy", "climate change policy", "AI safety"]

    generator.generate_dataset(
        topics=topics,
        samples_per_topic=args.samples,
        output_file=args.output,
        max_depth=args.depth,
        num_agents=args.agents,
        append=args.append,
    )


if __name__ == "__main__":
    main()