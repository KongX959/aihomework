import re

# ===================== 1. 变量与常量判定 =====================
def is_variable(s):
    """小写开头且长度<=2为变量，其余为常量"""
    if not isinstance(s, str) or not s: return False
    return s[0].islower() and len(s) <= 2

# ===================== 2. 核心工具函数 =====================
def parse_formula(s):
    s = s.strip()
    idx = s.find('(')
    if idx == -1: return s, []
    name, args_str = s[:idx], s[idx+1:-1]
    args, bracket, curr = [], 0, ''
    for char in args_str:
        if char == '(': bracket += 1
        elif char == ')': bracket -= 1
        if char == ',' and bracket == 0:
            args.append(curr.strip())
            curr = ''
        else: curr += char
    if curr: args.append(curr.strip())
    return name, args

def apply_sub(term, subs):
    if term in subs: return apply_sub(subs[term], subs)
    name, args = parse_formula(term)
    if not args: return term
    return f"{name}({','.join([apply_sub(a, subs) for a in args])})"

def format_clause(clause):
    if not clause: return "[]"
    content = ", ".join(clause)
    # 单元素元组末尾加逗号 [cite: 99, 100]
    return f"({content},)" if len(clause) == 1 else f"({content})"

# ===================== 3. MGU 最一般合一算法 =====================
def MGU(f1, f2):
    def unify(u, v, subs):
        u, v = apply_sub(u, subs), apply_sub(v, subs)
        if u == v: return subs
        if is_variable(u): return {**subs, u: v}
        if is_variable(v): return {**subs, v: u}
        u_n, u_a = parse_formula(u)
        v_n, v_a = parse_formula(v)
        if u_n != v_n or len(u_a) != len(v_a): return None
        for a, b in zip(u_a, v_a):
            subs = unify(a, b, subs)
            if subs is None: return None
        return subs
    return unify(f1.lstrip('~'), f2.lstrip('~'), {})

# ===================== 4. 一阶逻辑归结 ResolutionFOL =====================
def ResolutionFOL(KB):
    # 将初始 KB 存入列表，每项为 list 格式 [cite: 26]
    clauses = [list(c) for c in KB]
    steps = []
    
    # 打印初始子句 [cite: 27]
    for i, c in enumerate(clauses, 1):
        steps.append(f"{i} (Init) {format_clause(c)}")

    # 记录已处理的子句，用于去重
    seen_clauses = {tuple(sorted(c)) for c in clauses}
    
    # 待处理索引队列
    queue = list(range(len(clauses)))
    
    while queue:
        # 单元优先策略：从队列中选出最短的子句索引进行处理
        queue.sort(key=lambda x: len(clauses[x]))
        i = queue.pop(0)
        c1 = clauses[i]
        
        # 与所有已知子句尝试归结
        for j in range(len(clauses)):
            c2 = clauses[j]
            if i == j: continue
            
            for idx1, l1 in enumerate(c1):
                for idx2, l2 in enumerate(c2):
                    # 符号相反且谓词相同
                    if (l1.startswith('~') != l2.startswith('~')) and \
                       (l1.lstrip('~').split('(')[0] == l2.lstrip('~').split('(')[0]):
                        
                        subs = MGU(l1, l2)
                        if subs is not None:
                            # 生成新子句并应用替换 [cite: 28]
                            res = [apply_sub(x, subs) for k, x in enumerate(c1) if k != idx1] + \
                                  [apply_sub(x, subs) for k, x in enumerate(c2) if k != idx2]
                            
                            # 排序去重
                            res = sorted(list(set(res)))
                            res_key = tuple(res)
                            
                            if res_key not in seen_clauses:
                                seen_clauses.add(res_key)
                                clauses.append(res)
                                new_idx = len(clauses) - 1
                                queue.append(new_idx)
                                
                                # 格式化输出归结步骤 [cite: 28]
                                step_num = len(steps) + 1
                                t1 = f"{i+1}{chr(97+idx1)}" if len(c1) > 1 else f"{i+1}"
                                t2 = f"{j+1}{chr(97+idx2)}" if len(c2) > 1 else f"{j+1}"
                                s_str = "{" + ", ".join(f"{k}={v}" for k, v in subs.items()) + "}" if subs else ""
                                
                                if not res:
                                    steps.append(f"{step_num} R[{t1},{t2}]{s_str} = []")
                                    return steps
                                steps.append(f"{step_num} R[{t1},{t2}]{s_str} = {format_clause(res)}")
    return steps

if __name__ == '__main__':
    # ===================== 测试用例 =====================
    KB = [
    ("On(tony, mike)",), 
    ("On(mike, john)",), 
    ("Green(tony)",), 
    ("~Green(john)",),
    ("~On(xx,yy)", "~Green(xx)", "Green(yy)")
]
    
    print("推理过程：")
    for row in ResolutionFOL(KB):
        print(row)