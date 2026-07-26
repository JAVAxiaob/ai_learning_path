# AutoGen 面试题汇总 (35题)

> 位置: 06-llm/autogen/doc/
> 配套文档: AutoGen-MultiAgent协作框架.md | AutoGen流程图详解.md | AutoGen性能优化重难点.md

---

## 📊 题目分布

| 类别 | 题数 | 出现频率 |
|-----|-----|---------|
| Agent基础架构 | 10题 | ⭐⭐⭐⭐⭐ |
| Multi-Agent协作 | 10题 | ⭐⭐⭐⭐⭐ |
| Tool Calling | 5题 | ⭐⭐⭐⭐ |
| Memory & 评估 | 5题 | ⭐⭐⭐⭐ |
| 生产环境 | 5题 | ⭐⭐⭐ |

---

## 一、Agent基础架构题（10题）

### Q1. 单Agent vs Multi-Agent 怎么选？优缺点对比

### Q2. Agent的4大核心能力：Tool/Memory/Planning/Evaluation 逐一说明

### Q3. ReAct vs Plan-and-Execute vs Reflexion 三种规划模式对比场景

### Q4. System Prompt 怎么写才能让Agent人设稳定不跑偏？

### Q5. 什么是UserProxyAgent？和普通AssistantAgent区别？

### Q6. Handoff转人工怎么做？触发条件有哪些？

### Q7. 为什么要用Pydantic做Tool参数校验？手写字典校验的坑

### Q8. Agent人设漂移问题：PM Agent写代码了怎么办？

### Q9. Tool调用安全风险Top 5：SQL注入/命令注入/PII泄露/Sandbox逃逸/越权怎么防护

### Q10. Agent测试怎么测？单测/集成/E2E三级策略

---

## 二、Multi-Agent协作题（10题）

### Q11. 三大协作模式：群聊RoundRobin/顺序工作流Sequential/等级模式Hierarchical 选型场景

### Q12. 群聊模式怎么选发言人？下一个发言的Agent选哪个最合理

### Q13. 终止条件termination_condition组合怎么写？TextMention+MaxMessage+Token的逻辑运算

### Q14. 多Agent任务分配策略：静态角色分工 vs 动态Swarm分派优缺点

### Q15. Agent对话之间怎么共享信息？Blackboard黑板模式的实现

### Q16. 多Agent一致性问题：两个Agent输出矛盾结论怎么办？

### Q17. 成本控制：5人团队50次LLM调用2.75美元，如何省到0.3美元

### Q18. 终止条件冲突：程序员说COMPLETE但评审还没PASS时，会不会误终止

### Q19. 多Agent死循环Top5原因和5层防护网

### Q20. 软件开发5人团队(PM/架构/程序员/测试/评审)哪些角色可以合并精简？

---

## 三、Tool Calling（5题）

### Q21. Tool Calling原理大揭密：LLM真的"执行"代码了？还是多轮对话协议？

### Q22. JSON Schema优化：相同LLM，为什么你写的Tool调用成功率只有30%，别人写的90%+

### Q23. JSON解析失败率50%+怎么办？三种修复机制

### Q24. Tool重试策略：指数退避+FallBack降级路径

### Q25. 哪些函数适合做成@Tool？不该做成Tool的典型反例

---

## 四、Memory & 评估（5题）

### Q26. Short/Long/Shared三层Memory各存什么？Token预算怎么分配？

### Q27. SlidingWindow vs SummaryBuffer vs RAG检索 三种记忆实现对比

### Q28. Agent评估怎么打分？不是分类任务的Label怎么评估

### Q29. RAGAS vs LLM-as-Judge vs Human Kappa一致性评估

### Q30. Lost-in-the-Middle现象在Agent里怎么表现？(中间上下文被忽略)

---

## 五、生产环境（5题）

### Q31. 生产级Multi-Agent系统架构：Redis队列+Worker Pool+分布式缓存怎么搭

### Q32. 可观测性：每单$0.3异常突然变$5怎么根因定位？监控6大指标

### Q33. 并发请求下Agent状态隔离：KV Cache状态冲突问题如何解

### Q34. 模型降级策略：GPT4o限流→4o-mini→本地Ollama三级降级

### Q35. AutoGen vs LangGraph vs CrewAI vs MetaGPT 框架选型对比