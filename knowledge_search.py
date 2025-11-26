import re
from typing import Dict, Tuple, List, Any

import jieba as jieba
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

from byd_doc_rag.doc_preprocess import DocumentProcess, DynamicSemanticChunker
from byd_doc_rag.emb import BGEEmbedding
from byd_doc_rag.llm import LLM

CORE_COMPONENTS_PROMPT = """
任务：你的任务是从指定文本片段中提取关键信息要素。此次提取的目的是协助发现隐含的、表明合规性违规的描述内容。关键信息要素包括但不限于核心描述性事件、关键制造技术、工艺方法以及相关的限制条件与约束要求。
输入：{document_chunk}
输出：
"""

REVIEW_QUERIES_PROMPT = """
任务：你的任务是根据当前审阅的文本及提供的核心描述性参考信息，生成相关的搜索。这些搜索需针对文本中可能存在合规性违规的领域，为后续检索原始法规文件以开展详细核查提供支持。不同搜索用回车符号（"\n"）分隔开。
输入：{document_chunk}
核心要素：{core_components}
搜索：
"""


def generate_review_queries(model, document_chunk: str) -> List[str]:
    """
    双阶段自动生成查询: 1待查文档核心信息元素解构. 2多维度风险探测与定制化查询生成
    :param model: LLM
    :param document_chunk: 待审查文档
    :return: 5个查询
    """
    core_prompt = CORE_COMPONENTS_PROMPT.format(document_chunk=document_chunk)
    core_response = model.predict(core_prompt)
    print("待审查文档核心要素：" + core_response)
    # 阶段2
    queries_prompt = REVIEW_QUERIES_PROMPT.format(document_chunk=document_chunk, core_components=core_response)
    queries_response = model.predict(queries_prompt)
    print("生成的查询：" + queries_response)
    queries = queries_response.split('\n')  # 返回‘’包括的内容
    return queries[:5]


KEY_INFO_EXTRACTION_PROMPT = """
你的任务是从查询语句中提取关键信息，并按三个不同优先级分类:
最高优先级（max）：最重要的核心概念或实体
中等优先级（mid）：重要的修饰词或限定条件
字面优先级（lit）：具体数值、标准或技术规格

查询语句：{query}
max：
mid：
lit：
"""


class KeyInfoExtractor:
    """查询中关键信息提取"""
    def __init__(self, llm):
        self.llm = llm

    def extract_key_info(self, query: str) -> Dict[str, Tuple[str, float]]:
        """
        提取查询 中的重要元素、信息,建立三级重要性分类层次
        :param query: 查询
        :return: 分级字典 {分级：（查询，权重）}
        """
        prompt = KEY_INFO_EXTRACTION_PROMPT.format(query=query)
        response = self.llm.predict(prompt)
        lines = response.strip().split('\n')
        key_info = {}
        weights = {'max': 0.5, 'mid': 0.3, 'lit': 0.2}
        # TODO: response内容
        for line in lines:
            if line.startswith("max:"):
                key_info['max'] = (line[4:].strip(), weights['max'])
            elif line.startswith("mid:"):
                key_info['mid'] = (line[4:].strip(), weights['mid'])
            elif line.startswith("lit:"):
                key_info['lit'] = (line[4:].strip(), weights['lit'])

        return key_info


def compute_document_length_factor(chunk_token_length: int, avg_token_length: int = 200) -> float:
    """
    计算文档长度自适应因子，调整不同长度文档的权重分配，确保长短文档都能得到公平的评分机会。
    :param chunk_token_length: 块长度
    :param avg_token_length: 平均块长度
    :return: 长度自适应因子
    """
    lambda_dl = (avg_token_length + chunk_token_length) / (2 * avg_token_length)
    return np.clip(lambda_dl, 0.3, 1.7)  # 限制因子范围，避免极端值影响


def compute_term_significance(term_freq: int, doc_length_factor: float) -> float:
    """
    术语重要性计算,防止高频术语过度影响评分。
    :param term_freq: 术语频率
    :param doc_length_factor: 长度自适应因子
    :return: 术语重要性
    """
    significance = (2 * term_freq * doc_length_factor) / (term_freq + 1)
    return significance


def compute_term_rarity(doc_freq: int, total_docs: int) -> float:
    """
    术语稀有度计算,稀有度越高的术语在检索中的权重越大
    :param doc_freq: 该块出现的频率
    :param total_docs: 块总数
    :return: 术语稀有度
    """
    rarity = np.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
    return rarity


def compute_coherence_index(term: str, chunk_tokens: str, window_size: int = 20) -> float:
    """
    连贯性指数评估,连贯性高的术语往往在文档的特定区域集中出现，表明其与文档主题的强相关性。
    :param term: 术语
    :param chunk: 块
    :param window_size: 窗口大小
    :return: 连贯性指数
    """
    chunk_length = len(chunk_tokens)
    # 块过短
    if chunk_length == 0 or window_size > chunk_length:
        return 0.0

    max_coherence = 0.0

    for i in range(0, chunk_length - window_size + 1, 5):
        window = chunk_tokens[i:i + window_size]
        term_count = window.count(term)

        if term_count > 0:
            coherence = (term_count * window_size) / chunk_length
            max_coherence = max(max_coherence, coherence)

    return max_coherence

# 初始化中文tokenizer（用于长度统计）
chinese_tokenizer = tiktoken.get_encoding("cl100k_base")  # 适配中文的token编码

class GKGRRetriever:
    """评分融合与检索"""
    def __init__(self,
                 knowledge_base: List[str],
                 embedding_model,
                 key_info_extractor: KeyInfoExtractor,
                 llm,
                 config: Dict[str, Any] = None):
        self.knowledge_base = knowledge_base
        self.embedding_model = embedding_model
        self.key_info_extractor = key_info_extractor
        self.llm = llm

        default_config = {
            "lambda_param": 0.5,
            "top_k": 5,
            "rerank_enabled": True,
            "query_expansion": True,
            "similarity_threshold": 0.1
        }
        self.config = {**default_config, **(config or {})}

        self.kb_embeddings = self._precompute_embeddings()

    def _precompute_embeddings(self) -> np.ndarray:
        """标准文档的知识库嵌入"""
        embeddings = self.embedding_model.encode(self.knowledge_base)
        return embeddings

    def retrieve_with_scores(self, query: str) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        根据分数检索
        :param query: 查询
        :return: k个知识块
        """
        query_embedding = self.embedding_model.encode([query])[0]
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        # 查询与所有知识库文档块相似性得分
        sentence_scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.kb_embeddings
        )[0]
        # 分级的关键信息
        key_info = self.key_info_extractor.extract_key_info(query)
        # 计算知识性权重得分
        knowledge_scores = self._compute_knowledge_scores(key_info)
        # 总得分
        final_scores = []
        for i in range(len(self.knowledge_base)):
            norm_sent = sentence_scores[i]
            norm_know = knowledge_scores[i] / max(knowledge_scores) if max(knowledge_scores) > 0 else 0

            final_score = (self.config["lambda_param"] * norm_know +
                           (1 - self.config["lambda_param"]) * norm_sent)
            final_scores.append(final_score)

        results_with_scores = []
        for i, final_score in enumerate(final_scores):
            if final_score > self.config["similarity_threshold"]:
                score_details = {
                    "sentence_score": float(sentence_scores[i]),
                    "knowledge_score": float(knowledge_scores[i]),
                    "final_score": float(final_score)
                }
                results_with_scores.append((self.knowledge_base[i], final_score, score_details))

        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        return results_with_scores[:self.config["top_k"]]

    def _compute_knowledge_scores(self, key_info: Dict[str, Tuple[str, float]]) -> List[float]:
        """
        计算知识性权重得分
        :param key_info: 查询的关键、分级信息
        :return:
        """
        scores = []
        total_docs = len(self.knowledge_base)  # 块总数
        # 第一步：预处理知识库所有文档（分词+token计数），缓存结果避免重复计算
        kb_cache = []
        for chunk in self.knowledge_base:
            # 中文分词（精确模式，保留完整术语）
            chunk_tokens = jieba.lcut(chunk.strip(), cut_all=False)
            # 计算文档token数（更精准反映中文长度）
            chunk_token_count = len(chinese_tokenizer.encode(chunk))
            kb_cache.append((chunk_tokens, chunk_token_count))

        # 知识库所有块的平均长度
        # 第二步：计算知识库所有文档的平均token数（用于长度因子）
        avg_token_count = sum([cache[1] for cache in kb_cache]) / total_docs if total_docs > 0 else 200

        # 第三步：遍历每个文档块计算知识得分
        for idx, (chunk_tokens, chunk_token_count) in enumerate(kb_cache):
            chunk_score = 0.0
            # 计算中文文档长度因子
            lambda_dl = compute_document_length_factor(chunk_token_count, avg_token_count)
            # 遍历查询中的三级优先级信息
            for priority, (info_text, weight) in key_info.items():
                if not info_text.strip():
                    continue
                # 中文分词：将查询中的要素文本拆分为术语（如“轻型汽车氮氧化物”→["轻型汽车", "氮氧化物"]）
                query_terms = jieba.lcut(info_text.strip(), cut_all=False)
                # 过滤停用词（可选，进一步提升精准度）
                stop_words = {"的", "和", "与", "及", "等", "为", "是"}
                query_terms = [term for term in query_terms if term not in stop_words and len(term) >= 2]

                for term in query_terms:
                    if term in chunk_tokens:
                        # 1. 计算术语在当前文档块中的频率
                        term_freq = chunk_tokens.count(term)
                        if term_freq == 0:
                            continue  # 术语未出现，跳过
                        # 2. 计算术语重要性
                        significance = compute_term_significance(term_freq, lambda_dl)
                        # 3. 计算术语稀有度（统计包含该术语的文档块数量）
                        doc_freq = sum([1 for (cache_tokens, _) in kb_cache if term in cache_tokens])
                        rarity = compute_term_rarity(doc_freq, len(self.knowledge_base))
                        # 4. 计算术语连贯性（基于分词后的token列表）
                        coherence = compute_coherence_index(term, chunk_tokens)
                        # 5. 计算单术语得分（优先级权重×重要性×稀有度×(1+连贯性)）
                        term_score = significance * rarity * (1 + coherence) * weight
                        chunk_score += term_score

            scores.append(chunk_score)
        return scores

    def retrieve(self, query: str) -> Tuple[List[str], str]:
        """
        检索、重排序、增强查询
        :param query: 查询
        :return:
        """
        results_with_scores = self.retrieve_with_scores(query)

        documents = [doc for doc, _, _ in results_with_scores]
        # 根据查询、文档 重排序
        if self.config["rerank_enabled"] and len(documents) > 1:
            documents = self._llm_rerank(query, documents)
        # 根据查询、文档 增强查询
        augmented_query = query
        if self.config["query_expansion"]:
            augmented_query = self._augment_query(query, documents[:3])

        return documents, augmented_query

    def _llm_rerank(self, query: str, documents: List[str]) -> List[str]:
        """重排序优化"""
        if len(documents) <= 1:
            return documents

        rerank_prompt = f"""
任务：以下列出了若干中文文档（标准条款/技术规范），每个文档均标注了对应编号。同时提供了一个中文审查查询，请你根据文档与查询的相关性，对所有文档编号进行排序（从最相关到最不相关）。必须包含每一个文档编号，且每个编号仅出现一次，不得遗漏或重复。

排序核心依据：
1. 中文术语匹配度：文档是否包含查询中的核心分词术语（如“氮氧化物”“GB 18352.6-2016”）及同义词/近义词（如“排放限值”与“排放要求”）；
2. 术语权重契合度：核心术语（max 级，如标准编号）、限定条件（mid 级，如车型）、具体参数（lit 级，如 30mg/km）的匹配完整性；
3. 语义连贯性：核心术语在文档中是否集中出现（契合函数连贯性评分逻辑），与查询主题关联是否紧密；
4. 领域相关性：文档内容是否聚焦汽车研发合规场景（如排放、工况、技术规范），避免无关领域干扰。

示例格式：
    文档 1：<文档 1 内容>
    文档 2：<文档 2 内容>
    文档 3：<文档 3 内容>
    审查查询：<查询内容>
    排序结果：3,1,2

现在给出实际的文档和审查查询：

"""
        for i, doc in enumerate(documents):
            rerank_prompt += f"文档 {i + 1}: {doc[:150]}...\n"

        rerank_prompt += f"问题: {query}\n答案:"

        try:
            response = self.llm.predict(rerank_prompt)
            order_nums = [int(x.strip()) - 1 for x in response.split(',')
                          if x.strip().isdigit() and 0 <= int(x.strip()) - 1 < len(documents)]

            reranked = [documents[i] for i in order_nums if i < len(documents)]

            # 添加遗漏的文档
            used_indices = set(order_nums)
            for i, doc in enumerate(documents):
                if i not in used_indices:
                    reranked.append(doc)

            return reranked[:len(documents)]
        except:
            return documents

    def _augment_query(self, original_query: str, top_results: List[str]) -> str:
        """查询增强"""
        if not top_results:
            return original_query

        document_list = ""
        for i, doc in enumerate(top_results):
            document_list += f"文档 {i + 1}: {doc[:100]}...\n"

        augment_prompt = f"""
任务：基于提供的所有参考文档（中文标准条款/技术规范），对原始查询进行合规性增强优化。需深度融合参考文档中的核心信息（如标准编号、技术参数、限定条件、专业术语），确保增强后的查询既能覆盖原始查询意图，又能精准匹配中文知识库中的合规依据，适配汽车研发领域合规审查场景。

增强要求：
1. 必须整合参考文档中的关键信息：优先补充标准编号（如GB 18352.6-2016）、具体技术参数（如排放限值30mg/km）、限定条件（如轻型汽车、WLTC工况）；
2. 术语使用符合中文汽车研发规范：保留“氮氧化物”“国六b标准”等中文专业术语，英文缩写需补充中文全称（如“NOx（氮氧化物）”），避免单独出现英文；
3. 结构清晰：增强查询为完整中文陈述句，突出合规核查核心维度（如“参数是否符合”“是否满足标准要求”）；
4. 不冗余：仅融合与原始查询强相关的信息，不添加无关内容，长度控制在100字以内；
5. 引用依据：若融合多个文档信息，需标注文档序号（如“依据文档1、2”），确保可追溯。

原始查询：{original_query}
参考文档：
{document_list}

增强后的查询：
"""

        try:
            augmented = self.llm.predict(augment_prompt)
            return augmented.strip()
        except:
            return original_query

    def _count_chinese_length(self, text: str):
        """统计中文字符"""
        cleaned_text = text.replace(" ", "").replace("\n", "").replace("\t", "")
        return len(cleaned_text)

class ErrorAnalyzer:
    """偏差检测分析"""

    def __init__(self, llm):
        self.llm = llm

    def analyze_errors(self, document_chunk: str, query: str, retrieved_knowledge: List[str]) -> Dict[str, Any]:
        analysis_prompt = f"""
任务：基于提供的审查查询和中文参考标准，对给定的待审查文档进行合规偏差分析。分析必须严格遵循参考标准要求，聚焦文档中与查询相关的核心描述，且需与中文知识评分逻辑（分词、术语权重、连贯性）保持一致，确保偏差点可被后续修订环节精准响应。

核心分析维度（与 _compute_knowledge_scores 函数适配）：
1. 中文术语匹配：文档是否包含查询核心分词术语（如“氮氧化物”“GB 18352.6-2016”“排放限值”）及同义词/近义词（需呼应术语扩展逻辑）；
2. 技术参数一致性：文档中的具体数值、标准编号、限定条件（如“30mg/km”“WLTC工况”）是否与参考标准完全一致；
3. 术语连贯性：核心术语在文档中是否集中出现（契合连贯性评分逻辑），表述是否逻辑连贯、无歧义；
4. 长度适配性：文档中合规相关描述的长度是否合理（参考 token 计数逻辑），无关键信息过度简略或冗余问题。

待审查文档：{document_chunk}
审查查询：{query}
参考标准（按优先级排序）：
{chr(10).join([f"{i + 1}. {ref}" for i, ref in enumerate(retrieved_knowledge)])}

分析输出要求：
1. 先给出“合规”或“不合规”的明确结论；
2. 若不合规，分点列出偏差（每点对应1个核心问题）：
   - 标注涉及的中文核心分词术语（如“氮氧化物排放限值”）；
   - 对比文档表述与参考标准的差异（需引用标准序号）；
   - 说明偏差对应的函数评分维度（如“术语频率不足”“参数与标准冲突”“连贯性不足”）；
3. 若合规，简要说明文档中核心术语、参数、连贯性与参考标准的匹配情况，佐证评分逻辑合理性；
4. 语言简洁专业，仅基于提供的文档和标准分析，不添加额外假设，避免英文缩写单独出现（需补充中文全称）。

分析结果：
"""

        analysis = self.llm.predict(analysis_prompt)

        return {
            "analysis": analysis,
            "reference_support": retrieved_knowledge
        }


class RevisionGenerator:
    """修订建议生成"""
    def __init__(self, llm):
        self.llm = llm

    def generate_revisions(self, document_chunk: str, analysis: Dict[str, Any]) -> Dict[str, str]:
        revision_prompt = f"""
任务：基于提供的合规偏差分析结果和对应参考标准，对给定的待审查文档进行修订优化。必须严格遵循参考标准的要求，确保修订后的文档完全符合合规规范；若待审查文档与分析结果、参考标准无差异，无需进行修订，直接说明“文档符合参考标准要求，无需修订”。

待审查文档：{document_chunk}
合规偏差分析：{analysis['analysis']}
参考标准：
{chr(10).join([f"- {ref}" for ref in analysis['reference_support']])}

修订要求：
1. 修订内容需精准对应偏差分析中的违规点，每处修改均需明确依据的参考标准核心信息（如标准编号、关键参数）；
2. 保留原文档的技术逻辑和表述风格，仅修改违规或不合规部分，不新增无关内容；
3. 术语使用符合汽车研发领域中文规范（如“氮氧化物”“国六b标准”“WLTC工况”），避免英文缩写单独出现（需补充中文全称，如“NOx（氮氧化物）”）；
4. 修订后的文档需语句通顺、逻辑连贯，可直接用于技术归档或审批流程；
5. 若无需修订，需简要说明文档符合哪些参考标准的关键要求，佐证合规性。

修订结果：
"""

        revision = self.llm.predict(revision_prompt)

        return {
            "original_text": document_chunk,
            "revision_suggestions": revision,
            "modified_regions": analysis.get("error_regions", []),
            "confidence": self._calculate_confidence(analysis)
        }

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        ref_count = len(analysis.get("reference_support", []))
        error_count = len(analysis.get("error_regions", []))

        confidence = min(0.9, 0.5 + (ref_count * 0.1) + (error_count * 0.05))
        return confidence


def complete_review_process(document_chunk: str,
                            gkgr_framework: GKGRRetriever,
                            error_analyzer: ErrorAnalyzer,
                            revision_generator: RevisionGenerator) -> Dict[str, Any]:
    """完整审查流程"""
    # 生成查询
    review_queries = generate_review_queries(gkgr_framework.llm, document_chunk)
    print("生成的查询：", review_queries)
    results = {}
    for query in review_queries[:3]:
        # 检索到的文档、增强的查询
        retrieved_docs, augmented_query = gkgr_framework.retrieve(query)
        print("检索的文档数：", len(retrieved_docs))
        print("增强的查询数：", len(augmented_query))
        knowledge_refs = retrieved_docs
        # 根据待审查文档、查询、检索到的标准文档分析误差
        analysis = error_analyzer.analyze_errors(document_chunk, query, knowledge_refs)
        # 根据待审查文档、分析报告生成修订建议
        revision = revision_generator.generate_revisions(document_chunk, analysis)

        results[query] = {
            "retrieved_knowledge": retrieved_docs,
            "augmented_query": augmented_query,
            "analysis": analysis,
            "revision": revision
        }

    return results


if __name__ == "__main__":
    # 初始化系统组件
    # 构造大模型
    print("="*10 + "构建大模型" + "="*10)
    llm = LLM(
        type='hunyuan',
        # type='qwen',
    )
    # 嵌入模型
    print("=" * 10 + "构建嵌入模型" + "=" * 10)
    # embedding = BGEEmbedding(model_name=r"C:\Users\chao\Downloads\bge-m3")
    embedding = BGEEmbedding(model_name=r"D:\移动硬盘\下载\all-MiniLM-L6-v2")
    # Query 分级工具
    key_extractor = KeyInfoExtractor(llm)
    # 知识库 构建工具
    processor = DocumentProcess()
    # 获取标准文档列表
    print("=" * 10 + "加载文档" + "=" * 10)
    documents = processor.load_documents(r"./docs")
    # 对文档进行动态语义分块
    chunker = DynamicSemanticChunker()
    knowledge_base = []
    print("=" * 10 + "分块文档" + "=" * 10)
    for doc in documents:
        chunks = chunker.split_text(doc)  # 分好块的字典
        knowledge_base.extend(chunks.values())

    print("=" * 10 + "初始化检索器、分析器、修订器" + "=" * 10)
    # 初始化检索器
    gkgr_retriever = GKGRRetriever(
        knowledge_base=knowledge_base,
        embedding_model=embedding,  # 传入BGEEmbedding实例而不是其内部的embed_model
        key_info_extractor=key_extractor,
        llm=llm
    )

    # 初始化分析器
    error_analyzer = ErrorAnalyzer(llm)
    revision_generator = RevisionGenerator(llm)

    # 待审查的文档内容
    sample_document = """
        轻型汽车发动机氮氧化物（NOx）排放限值设定为40mg/km，测试过程遵循NEDC循环工况，符合国六排放标准相关要求。该限值适用于额定功率≤160kW的汽油发动机，文档未明确对应具体标准编号及实施时间。
"""
    print("=" * 10 + "开始审查" + "=" * 10)
    # 执行审查
    result = complete_review_process(
        sample_document,
        gkgr_retriever,
        error_analyzer,
        revision_generator
    )
    print("=" * 10 + "审查结束" + "=" * 10)
    # 查看审查结果
    for query, analysis in result.items():
        print(f"审查问题: {query}")
        print(f"修订建议: {analysis['revision']['revision_suggestions']}")
        print("-" * 50)



