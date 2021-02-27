# 头脑风暴
## 简化问题
1. 疯癫的海盗
问题：
有$n$个海盗，分赃$N$个钱币。遵循以下的规则，海盗中的高级首领，会提出一个分赃方案，供所有海盗表决，如果获得至少一半的赞成票，那方案通过，否则该首领会被抛入大海，胜率海盗中威望最高的，再提出一个方案，重复上述过程，直至分赃成功。假设整个过程中，海盗们都是理性的人，即他们首要条件是想保证自己的存活，然后尽力争取最多的钱币，而且能够获得相同的钱币条件下，更加期望杀死其他海盗。请问最终的分赃方案是什么样子的。
分析：
我们可以从简单的情况开始分析这个问题。假如分赃100枚钱币，只有一个海盗，那么方案很显然。假如有两个海盗，海盗2的方案肯定可以通过（自己会投赞成票），那么海盗2肯定会把100枚钱币都分给自己；假如有三个海盗，海盗3知道如果自己被否决，那么海盗1也将一无所获，所以海盗3只要提出一个方案，分给自己99枚，分给海盗1一枚即可。以此类推，我们会发现如果$n=2k+1$，那么他只需要给$1,3,...$的海盗每人一枚，剩余的都留给自己；如果$n=2k$，那么他只需要给$2,4,...$的海盗每人一枚，剩余留给自己。
拓展：
- 如果方案提出者不能参加投票呢？
- 如果要超过一半的人才能成功呢？

2. 老虎和绵羊
问题：
假设100只老虎和一只绵羊在草原上，老虎特别想吃掉这只绵羊，但是一旦某只老虎吃掉绵羊后，他自己就会变成绵羊。假设老虎都是很理性的，即都想先保证自己的存活，在考虑吃。请问这只绵羊最终会被吃掉吗？
分析：
