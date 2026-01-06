import re
from collections import Counter

def enhanced_normalize_answer(s):
    """更宽松的答案标准化方法"""
    if s is None:
        return ""
    
    # 处理列表类型的答案
    if isinstance(s, list):
        s = s[0] if len(s) > 0 else ""
    
    # 移除所有标点和特殊字符
    s = re.sub(r'[^\w\s]', '', str(s))
    # 转换为小写
    s = s.lower()
    # 移除多余空格
    s = ' '.join(s.split())
    # 处理数字格式
    s = re.sub(r'\b(\d+),(\d+)\b', r'\1\2', s)  # 移除千分位逗号
    # 统一单位缩写
    unit_map = {'km': 'kilometer', 'mi': 'mile', 'kg': 'kilogram', 'lb': 'pound'}
    for short, full in unit_map.items():
        s = s.replace(short, full)
    return s

def enhanced_f1_score(prediction, ground_truth):
    """增强版F1计算，支持部分匹配"""
    pred_tokens = enhanced_normalize_answer(prediction).split()
    gold_tokens = enhanced_normalize_answer(ground_truth).split()
    
    # 处理空答案
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # 计算最长公共子序列(LCS)作为部分匹配
    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if X[i-1] == Y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def enhanced_exact_match(prediction, ground_truth, threshold=0.9):
    """增强版精确匹配，支持相似度阈值"""
    normalized_pred = enhanced_normalize_answer(prediction)
    normalized_gold = enhanced_normalize_answer(ground_truth)
    
    if normalized_pred == normalized_gold:
        return 1.0
    
    # 计算Jaccard相似度
    pred_set = set(normalized_pred.split())
    gold_set = set(normalized_gold.split())
    intersection = pred_set & gold_set
    union = pred_set | gold_set
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # 计算编辑距离相似度
    def edit_similarity(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 or j == 0:
                    dp[i][j] = max(i, j)
                else:
                    cost = 0 if a[i-1] == b[j-1] else 1
                    dp[i][j] = min(dp[i-1][j] + 1,   # 删除
                                   dp[i][j-1] + 1,   # 插入
                                   dp[i-1][j-1] + cost)  # 替换
        
        max_len = max(m, n)
        return 1 - (dp[m][n] / max_len) if max_len > 0 else 1.0
    
    edit_sim = edit_similarity(normalized_pred, normalized_gold)
    
    # 综合相似度
    similarity = max(jaccard, edit_sim, enhanced_f1_score(prediction, ground_truth))
    return 1.0 if similarity >= threshold else 0.0

def analyze_discrepancies(pred_answers, gold_answers, sample_size=50):
    sample_size=len(pred_answers)
    """分析预测与标准答案的差异（直接使用数组）"""
    discrepancies = []
    for i in range(min(sample_size, len(gold_answers))):
        pred = pred_answers[i]
        gold = gold_answers[i]
        
        # 标准化答案
        norm_pred = enhanced_normalize_answer(pred)
        norm_gold = enhanced_normalize_answer(gold)
        
        # 计算相似度
        em = enhanced_exact_match(pred, gold)
        f1 = enhanced_f1_score(pred, gold)
        
        # 记录差异
        if em < 0.9 or f1 < 0.8:
            discrepancies.append({
                'id': i,  # 使用索引作为ID
                'pred': pred,
                'gold': gold,
                'norm_pred': norm_pred,
                'norm_gold': norm_gold,
                'em': em,
                'f1': f1
            })
    
    # 输出分析报告
    if discrepancies:
        print(f"\n发现 {len(discrepancies)}/{min(sample_size, len(gold_answers))} 个差异样本")
        print("常见差异类型:")
        
        # 统计差异原因
        reasons = Counter()
        for d in discrepancies:
            if d['norm_gold'] in d['norm_pred']:
                reasons['多余内容'] += 1
            elif d['norm_pred'] in d['norm_gold']:
                reasons['信息不全'] += 1
            elif set(d['norm_pred'].split()) & set(d['norm_gold'].split()):
                reasons['部分匹配'] += 1
            else:
                reasons['完全不符'] += 1
        
        for reason, count in reasons.most_common():
            print(f"- {reason}: {count} 样本")
        
        # 打印典型示例
        # print("\n典型差异示例:")
        # for i, d in enumerate(discrepancies[:3]):
        #     print(f"\n示例 {i+1} (索引: {d['id']}):")
        #     print(f"预测: {d['pred']} (标准化: {d['norm_pred']})")
        #     print(f"标准: {d['gold']} (标准化: {d['norm_gold']})")
        #     print(f"EM: {d['em']:.2f}, F1: {d['f1']:.2f}")
    else:
        print("\n未发现显著差异样本")
    
    return discrepancies

def enhanced_hotpot_eval(pred_answers, gold_answers):
    """增强版HotpotQA评估（直接使用数组）"""
    total_em = 0
    total_f1 = 0
    count = 0
    
    # 确保我们有相同数量的预测和答案
    n = min(len(pred_answers), len(gold_answers))
    
    for i in range(n):
        pred = pred_answers[i]
        gold = gold_answers[i]
        
        # 处理列表类型的答案
        if isinstance(gold, list) and len(gold) > 0:
            gold = gold[0]
        
        # 计算增强指标
        em = enhanced_exact_match(pred, gold)
        f1 = enhanced_f1_score(pred, gold)
        
        total_em += em
        total_f1 += f1
        count += 1
    
    # 分析差异
    if count > 0:
        try:
            analyze_discrepancies(pred_answers[:n], gold_answers[:n])
        except Exception as e:
            print(f"差异分析失败: {str(e)}")
    
    return {
        'em': total_em / count if count > 0 else 0,
        'f1': total_f1 / count if count > 0 else 0,
        'count': count
    }