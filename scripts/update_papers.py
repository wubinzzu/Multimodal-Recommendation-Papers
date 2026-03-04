import arxiv
import json
import re
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from zai import ZhipuAiClient

class PaperUpdater:
    def __init__(self, paper_path):
        self.existing_papers = set()
        self.keywords = self.load_keywords()
        self.paper_path = paper_path
        self.client = self.get_paper_classify_agent()

    def load_keywords(self) -> List[str]:
        """加载查询关键词"""
        with open('scripts/keywords.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def get_paper_classify_agent(self):
        """初始化GLM API客户端"""
        key = os.environ.get("ZHIPUAI_API_KEY")
        if not key:
            print("API key Missing")
            exit(0)
        return ZhipuAiClient(api_key=key)# 请替换为您的API Key
        

    def classify_paper_with_llm(self, paper_info: Dict) -> Tuple[bool, str, str]:
        """使用GLM API对论文进行分类"""
        system_prompt = """
你是一位专注于推荐系统领域的资深论文分类专家。

为了更准确地判断论文是否属于“多模态推荐系统”领域，请参考以下背景知识：

多模态推荐系统是指利用多种模态数据（如文本、图像、音频、视频等）来丰富物品表示或用户偏好，从而提升推荐准确性和可解释性的研究方向。核心在于处理异构数据源并进行模态融合或对齐。

典型特征包括但不限于：
1. 显式利用多源内容特征，例如同时使用商品的文本描述（标题/评论）和视觉特征（图片/视频缩略图）；
2. 涉及模态融合机制，将不同模态的特征向量映射到同一空间或进行交互建模；
3. 使用预训练模型（如ResNet, ViT, BERT, CLIP）提取多模态特征并应用于推荐任务；
4. 关注跨模态对齐、模态缺失处理或多模态图神经网络；
5. 视觉感知推荐、跨模态哈希推荐等。

不属于多模态推荐的情况包括：
- 仅使用ID特征的传统协同过滤（未利用内容特征）；
- 仅使用单一模态特征的推荐（如纯文本推荐或纯图像处理），未涉及多模态融合；
- 纯粹的社交推荐或序列推荐，除非文中明确结合了多模态内容信息；
- 通用的计算机视觉或多模态学习任务，不涉及推荐系统应用。

如果论文明确结合了两种或以上的模态信息来改进推荐系统，请将其归类为“多模态推荐”。
"""
        prompt = f"""
请对以下学术论文进行分类。论文信息：

标题：{paper_info['title']}
摘要：{paper_info['summary']}

请判断该论文是否属于多模态推荐系统领域。

请严格按以下格式返回结果：

是否属于多模态推荐：是/否

其中冒号前的内容为固定中文字符串，冒号后的内容为分类结果。
不要输出多余信息。
因为后续处理流程会使用 line.split('：')[1].strip()=='是' 来判断是否属于多模态推荐。
"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4.5-air",
                messages=[
                    {"role": "system", "content": system_prompt},  # 添加System消息
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10000,
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            #print("GLM API返回结果:", result)
            return ("是" == result.split('：')[1].strip())
            
        except Exception as e:
            print(f"GLM API调用失败: {e}")
            return False

    def load_existing_papers(self) -> None:
        """加载已有论文信息用于去重"""
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        arxiv_pattern = r'arxiv\.org/abs/(\d+\.\d+)'
        self.existing_papers = set(re.findall(arxiv_pattern, content, re.IGNORECASE))

    def query_new_papers(self) -> List[Dict]:
        """查询arXiv最新论文"""
        new_papers = []
        client = arxiv.Client()
        
        for keyword in self.keywords:
            search_query = f'ti:"{keyword}" OR abs:"{keyword}"'
            search = arxiv.Search(
                query=search_query,
                max_results=200,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            try:
                for result in client.results(search):
                    # 日期过滤：只看100天内的
                    if result.published.date() < (datetime.now() - timedelta(days=100)).date():
                        continue
                    
                    arxiv_id = result.entry_id.split('/')[-1]
                    for i in range(1,10):
                        arxiv_id = arxiv_id.replace(f'v{i}','')
                    if arxiv_id in self.existing_papers:
                        continue

                    # 构建论文信息字典，补充 journal_ref 和 comment
                    paper_info = {
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'arxiv_id': arxiv_id,
                        'pdf_url': result.pdf_url,
                        'year': result.published.year,
                        'summary': result.summary,
                        'primary_category': str(result.primary_category),
                        'journal_ref': result.journal_ref,  # 新增
                        'comment': result.comment            # 新增
                    }
                    paper_info['venue'] = str(self.determine_venue(paper_info)).strip()
                    if paper_info['venue'] == 'Arxiv':
                        continue
                    # 使用GLM进行分类
                    is_right = self.classify_paper_with_llm(paper_info)
                    
                    if is_right:
                        new_papers.append(paper_info)
                        self.existing_papers.add(arxiv_id)

                    
            except Exception as e:
                print(f"查询关键词 '{keyword}' 时出错: {e}")
                continue
                
        return new_papers

    def format_paper_entry(self, paper: Dict) -> str:
        """格式化论文条目"""
        year = paper['year']
        venue = paper['venue']
        
        if len(venue.split(' ')) > 1:
            venue,year = venue.split(' ')
        abs_url = paper['pdf_url'].replace('pdf','abs')
        for i in range(1,10):
            abs_url = abs_url.replace(f'v{i}','')
        entry = f"- `{venue}({year})`{paper['title']} **[[PDF]({abs_url})]**\n"
        
        return entry
    def determine_venue(self, paper: Dict) -> str:
        target_venues = [
                "NeurIPS", "ICML", "ICLR", "AAAI", "IJCAI", "TOIS", "NAACL"
                "ACL", "EMNLP", "WSDM", "TMLR", "GenRec", "ICDE"
                "KDD", "WWW", "SIGIR-AP","SIGIR", "TKDE", "TORS", "CIKM", "RecSys",
                "JMLR", "TPAMI", "TIP", "NIPS", "EMNLP"
            ]
        def extract_venue(desc):
            if not desc:
                return 'Arxiv'
            for venue in target_venues:
                if venue.lower() in desc.lower():
                    # 尝试提取年份
                    # 正则逻辑：查找 Venue 名称后的年份
                    year_match = re.search(rf'{venue}.*?((?:19|20)\d{{2}})', desc, re.IGNORECASE)
                    if year_match:
                        return f"{venue} {year_match.group(1)}"
                    # 如果没找到年份，仅返回会议名
                    return venue
            return 'Arxiv'
        """根据论文信息确定会议/期刊"""
        # 1. 检查是否有正式的期刊引用
        if paper.get('journal_ref'):
            venue = extract_venue(paper['journal_ref'].strip())
            if venue != 'Arxiv':
                return venue
        
        # 2. 检查 Comment 字段中的会议信息
        comment = paper.get('comment', '')
        return extract_venue(comment)

    def update_readme(self, new_papers: List[Dict]) -> bool:
        """更新主README.md文件"""
        if not new_papers:
            print("没有发现新论文")
            return False

        # 读取现有README内容
        with open(self.paper_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 找到Generative Recommendation部分
        pattern = r'(### Multi-Modal Recommendation System\n)(.*?)(?=\n###|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("未找到标题部分")
            return False

        # 构建新内容
        new_entries = []
        for paper in new_papers:
            new_entries.append(self.format_paper_entry(paper))

        updated_section = match.group(1) + ''.join(new_entries) + match.group(2)
        new_content = content.replace(match.group(0), updated_section)

        # 写回文件
        with open(self.paper_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"成功添加 {len(new_papers)} 篇新论文")
        return True

    def main(self):
        """主执行函数"""
        self.load_existing_papers()
        new_papers = self.query_new_papers()
        
        if self.update_readme(new_papers):
            commit_message = f"Auto-update: Add {len(new_papers)} new papers - {datetime.now().strftime('%Y-%m-%d')}"
            print(commit_message)
        else:
            print("无需更新")

if __name__ == "__main__":
    updater = PaperUpdater(paper_path='README.md')
    updater.main()
