# 向量数据库
向量数据库（Vector Database），也叫矢量数据库，主要用来存储和处理向量数据。

在数学中，向量是有大小和方向的量，可以使用带箭头的线段表示，箭头指向即为向量的方向，线段的长度表示向量的大小。两个向量的距离或者相似性可以通过欧式距离或者余弦距离得到。

图像、文本和音视频这种非结构化数据都可以通过某种变换或者嵌入学习转化为向量数据存储到向量数据库中，从而实现对图像、文本和音视频的相似性搜索和检索。这意味着您可以使用向量数据库根据语义或上下文含义查找最相似或相关的数据，而不是使用基于精确匹配或预定义标准查询数据库的传统方法。

向量数据库的主要特点是高效存储与检索。利用索引技术和向量检索算法能实现高维大数据下的快速响应。向量数据库也是一种数据库，除了要管理向量数据外，还是支持对传统结构化数据的管理。实际使用时，有很多场景会同时对向量字段和结构化字段进行过滤检索，这对向量数据库来说也是一种挑战。

## <font style="color:rgb(51, 51, 51);">向量嵌入(Vector Embeddings)</font>
<font style="color:rgb(51, 51, 51);">对于传统数据库，搜索功能都是基于不同的索引方式（B Tree、倒排索引等）加上精确匹配和排序算法（BM25、TF-IDF）等实现的。本质还是基于文本的精确匹配，这种索引和搜索算法对于关键字的搜索功能非常合适，但对于语义搜索功能就非常弱。</font>

<font style="color:rgb(51, 51, 51);">例如，如果你搜索 “小狗”，那么你只能得到带有“小狗” 关键字相关的结果，而无法得到 “柯基”、“金毛” 等结果，因为 “小狗” 和“金毛”是不同的词，传统数据库无法识别它们的语义关系，所以传统的应用需要人为的将 “小狗” 和“金毛”等词之间打上特征标签进行关联，这样才能实现语义搜索。而如何将生成和挑选特征这个过程，也被称为 Feature Engineering (特征工程)，它是将原始数据转化成更好的表达问题本质的特征的过程。</font>

<font style="color:rgb(51, 51, 51);">但是如果你需要处理非结构化的数据，就会发现非结构化数据的特征数量会开始快速膨胀，例如我们处理的是图像、音频、视频等数据，这个过程就变得非常困难。例如，对于图像，可以标注颜色、形状、纹理、边缘、对象、场景等特征，但是这些特征太多了，而且很难人为的进行标注，所以我们需要一种自动化的方式来提取这些特征，而这可以通过 Vector Embedding 实现。</font>

<font style="color:rgb(51, 51, 51);">Vector Embedding 是由 AI 模型（例如大型语言模型 LLM）生成的，它会根据不同的算法生成高维度的向量数据，代表着数据的不同特征，这些特征代表了数据的不同维度。例如，对于文本，这些特征可能包括词汇、语法、语义、情感、情绪、主题、上下文等。对于音频，这些特征可能包括音调、节奏、音高、音色、音量、语音、音乐等。</font>

<font style="color:rgb(51, 51, 51);">例如对于目前来说，文本向量可以通过 OpenAI 的 text-embedding-ada-002 模型生成，图像向量可以通过 clip-vit-base-patch32 模型生成，而音频向量可以通过 wav2vec2-base-960h 模型生成。这些向量都是通过 AI 模型生成的，所以它们都是具有语义信息的。</font>

<font style="color:rgb(51, 51, 51);">例如我们将这句话 “Your text string goes here” 用 text-embedding-ada-002 模型进行文本 Embedding，它会生成一个 1536 维的向量，得到的结果是这样：</font>`<font style="color:rgb(51, 51, 51);background-color:rgb(243, 244, 244);">“-0.006929283495992422, -0.005336422007530928, ... -4547132266452536e-05,-0.024047505110502243”</font>`<font style="color:rgb(51, 51, 51);">，它是一个长度为 1536 的数组。这个向量就包含了这句话的所有特征，这些特征包括词汇、语法，我们可以将它存入向量数据库中，以便我们后续进行语义搜索。</font>

## <font style="color:rgb(51, 51, 51);">特征和向量</font>
<font style="color:rgb(51, 51, 51);">虽然向量数据库的核心在于相似性搜索 (Similarity Search)，但在深入了解相似性搜索前，我们需要先详细了解一下特征和向量的概念和原理。</font>

<font style="color:rgb(51, 51, 51);">我们先思考一个问题？为什么我们在生活中区分不同的物品和事物？</font>

<font style="color:rgb(51, 51, 51);">如果从理论角度出发，这是因为我们会通过识别不同事物之间不同的特征来识别种类，例如分别不同种类的小狗，就可以通过体型大小、毛发长度、鼻子长短等特征来区分。如下面这张照片按照体型排序，可以看到体型越大的狗越靠近坐标轴右边，这样就能得到一个体型特征的一维坐标和对应的数值，从 0 到 1 的数字中得到每只狗在坐标系中的位置。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729392475132-5a9b671f-98a3-4891-90d8-9c51d8aa80f3.png)

<font style="color:rgb(51, 51, 51);">然而单靠一个体型大小的特征并不够，像照片中哈士奇、金毛和拉布拉多的体型就非常接近，我们无法区分。所以我们会继续观察其它的特征，例如毛发的长短。</font>

![](https://cdn.nlark.com/yuque/0/2024/jpeg/2424104/1729395865931-b2199ee4-23e2-409d-baac-7c7ef38ab125.jpeg)

<font style="color:rgb(51, 51, 51);">这样每只狗对应一个二维坐标点，我们就能轻易的将哈士奇、金毛和拉布拉多区分开来，如果这时仍然无法很好的区分德牧和罗威纳犬。我们就可以继续再从其它的特征区分，比如鼻子的长短，这样就能得到一个三维的坐标系和每只狗在三维坐标系中的位置。</font>

<font style="color:rgb(51, 51, 51);">在这种情况下，只要特征足够多，就能够将所有的狗区分开来，最后就能得到一个高维的坐标系，虽然我们想象不出高维坐标系长什么样，但是在数组中，我们只需要一直向数组中追加数字就可以了。</font>

<font style="color:rgb(51, 51, 51);">实际上，只要维度够多，我们就能够将所有的事物区分开来，世间万物都可以用一个多维坐标系来表示，它们都在一个高维的特征空间中对应着一个坐标点。</font>

<font style="color:rgb(51, 51, 51);">那这和相似性搜索 (Similarity Search) 有什么关系呢？你会发现在上面的二维坐标中，德牧和罗威纳犬的坐标就非常接近，这就意味着它们的特征也非常接近。我们都知道向量是具有大小和方向的数学结构，所以可以将这些特征用向量来表示，这样就能够通过计算向量之间的距离来判断它们的相似度，这就是</font>**<font style="color:rgb(51, 51, 51);">相似性测量</font>**<font style="color:rgb(51, 51, 51);">。</font>

## <font style="color:rgb(51, 51, 51);">相似性测量 (Similarity Measurement)</font>
<font style="color:rgb(51, 51, 51);">上面我们讨论了向量数据库的不同搜索算法，但是还没有讨论如何衡量相似性。在相似性搜索中，需要计算两个向量之间的距离，然后根据距离来判断它们的相似度。</font>

<font style="color:rgb(51, 51, 51);">而如何计算向量在高维空间的距离呢？有三种常见的向量相似度算法：欧几里德距离、余弦相似度和点积相似度。</font>

### <font style="color:rgb(51, 51, 51);">欧几里得距离（Euclidean Distance）</font>
欧几里得距离是用于衡量两个点（或向量）之间的直线距离的一种方法。它是最常用的距离度量之一，尤其在几何和向量空间中。可以将其理解为两点之间的“直线距离”。

<font style="color:rgb(51, 51, 51);">其中，A 和 B 分别表示两个向量，</font>_<font style="color:rgb(51, 51, 51);">n</font>_<font style="color:rgb(51, 51, 51);"> 表示向量的维度。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729608492126-1a8a5656-d545-4cc2-b986-46016ad7686d.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729395294470-1651a4e4-587b-4457-9a77-7e090a84d6db.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729609098685-0522b181-7435-42aa-a5c2-5123f0b0b19e.png)

<font style="color:rgb(51, 51, 51);">欧几里得距离算法的优点是可以反映向量的绝对距离，适用于需要考虑向量长度的相似性计算。例如推荐系统中，需要根据用户的历史行为来推荐相似的商品，这时就需要考虑用户的历史行为的数量，而不仅仅是用户的历史行为的相似度。</font>

### <font style="color:rgb(51, 51, 51);">余弦相似度（Cosine Similarity）</font>
<font style="color:rgb(51, 51, 51);">余弦相似度（Cosine Similarity）是一种用于衡量两个向量之间相似度的指标。它通过计算两个向量夹角的余弦值来判断它们的相似程度。其值介于 -1 和 1 之间，通常用于文本分析和推荐系统等领域。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729609456324-050edbf6-d233-47a3-9baf-c9077c2a2d0f.png)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729395305603-2de8d99b-bb5d-44a3-a9fb-8b3e7402d4b6.png)

**<font style="color:rgb(36, 41, 47);">特点</font>**

<font style="color:rgb(36, 41, 47);">值域：结果在 -1 到 1 之间。1 表示完全相似，0 表示不相关，-1 表示完全相反。</font>

<font style="color:rgb(51, 51, 51);">余弦相似度对向量的长度不敏感，只关注向量的方向，因此适用于高维向量的相似性计算。例如语义搜索和文档分类。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729609564551-98a8dd4e-ee74-4471-b184-807098cfd047.png)

## <font style="color:rgb(51, 51, 51);">相似性搜索 (Similarity Search)</font>
<font style="color:rgb(51, 51, 51);">既然我们知道了可以通过比较向量之间的距离来判断它们的相似度，那么如何将它应用到真实的场景中呢？如果想要在一个海量的数据中找到和某个向量最相似的向量，我们需要对数据库中的每个向量进行一次比较计算，但这样的计算量是非常巨大的，所以我们需要一种高效的算法来解决这个问题。</font>

<font style="color:rgb(51, 51, 51);">高效的搜索算法有很多，其主要思想是通过两种方式提高搜索效率：</font>

<font style="color:rgb(51, 51, 51);">1）</font>**<font style="color:rgb(51, 51, 51);">减少向量大小</font>**<font style="color:rgb(51, 51, 51);">——通过降维或减少表示向量值的长度。</font>

<font style="color:rgb(51, 51, 51);">2）</font>**<font style="color:rgb(51, 51, 51);">缩小搜索范围</font>**<font style="color:rgb(51, 51, 51);">——可以通过聚类或将向量组织成基于树形、图形结构来实现，并限制搜索范围仅在最接近的簇中进行，或者通过最相似的分支进行过滤。</font>

<font style="color:rgb(51, 51, 51);">我们首先来介绍一下大部分算法共有的核心概念，也就是聚类。</font>

### <font style="color:rgb(51, 51, 51);">K-Means </font>
<font style="color:rgb(51, 51, 51);">我们可以在保存向量数据后，先对向量数据先进行聚类。例如下图在二维坐标系中，划定了 4 个聚类中心，然后将每个向量分配到最近的聚类中心，经过聚类算法不断调整聚类中心位置，这样就可以将向量数据分成 4 个簇。每次搜索时，只需要先判断搜索向量属于哪个簇，然后再在这一个簇中进行搜索，这样就从 4 个簇的搜索范围减少到了 1 个簇，大大减少了搜索的范围。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729395327042-e583bcb3-7fda-483f-8aba-586eb587d2af.png)

**工作原理**

1. **初始化**：选择 $ k $ 个随机点作为初始聚类中心。
2. **分配**：将每个向量分配给离它最近的聚类中心，形成 $ k $ 个簇。
3. **更新**：重新计算每个簇的聚类中心，即簇内所有向量的平均值。
4. **迭代**：重复“分配”和“更新”步骤，直到聚类中心不再变化或达到最大迭代次数。

**举例说明**

假设我们有一组二维向量（点），例如：$ [(1, 2), (2, 1), (4, 5), (5, 4), (8, 9)] $，我们希望将它们聚成四个簇（$ k = 4 $）。

1. **初始化**：随机选择四个点作为初始聚类中心，比如 $ (1, 2) $、$ (2, 1) $、$ (4, 5) $ 和 $ (8, 9) $。
2. **分配**：计算每个点到四个聚类中心的距离，将每个点分配给最近的聚类中心。

形成四个簇：$ [(1, 2)] $，$ [(2, 1)] $，$ [(4, 5), (5, 4)] $，$ [(8, 9)] $。

    - 点 $ (1, 2) $ 更接近 $ (1, 2) $。
    - 点 $ (2, 1) $ 更接近 $ (2, 1) $。
    - 点 $ (4, 5) $ 更接近 $ (4, 5) $。
    - 点 $ (5, 4) $ 更接近 $ (4, 5) $。
    - 点 $ (8, 9) $ 更接近 $ (8, 9) $。
3. **更新**：计算每个簇的新聚类中心。
    - 第一个簇的聚类中心是 $ (1, 2) $。
    - 第二个簇的聚类中心是 $ (2, 1) $。
    - 第三个簇的新聚类中心是 $ ((4+5)/2, (5+4)/2) = (4.5, 4.5) $。
    - 第四个簇的聚类中心是 $ (8, 9) $。
4. **迭代**：重复分配和更新步骤，直到聚类中心不再变化。

**优点**：通过将向量聚类，我们可以在进行相似性搜索时只需计算查询向量与每个聚类中心之间的距离，而不是与所有向量计算距离。这大大减少了计算量，提高了搜索效率。这种方法在处理大规模数据时尤其有效，可以显著加快搜索速度。

**<font style="color:rgb(51, 51, 51);">缺点</font>**<font style="color:rgb(51, 51, 51);">，例如在搜索的时候，如果搜索的内容正好处于两个分类区域的中间，就很有可能遗漏掉最相似的向量。</font>

### <font style="color:rgb(51, 51, 51);">Hierarchical Navigable Small Worlds (HNSW)</font>
<font style="color:rgb(51, 51, 51);">除了聚类以外，也可以通过构建树或者构建图的方式来实现近似最近邻搜索。这种方法的基本思想是每次将向量加到数据库中的时候，就先找到与它最相邻的向量，然后将它们连接起来，这样就构成了一个图。当需要搜索的时候，就可以从图中的某个节点开始，不断的进行最相邻搜索和最短路径计算，直到找到最相似的向量。</font>

<font style="color:rgb(51, 51, 51);">这种算法能保证搜索的质量，但是如果图中所以的节点都以最短的路径相连，如图中最下面的一层，那么在搜索的时候，就同样需要遍历所有的节点。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729395351921-631a694f-0002-4b3e-ba31-567cd629829d.png)

<font style="color:rgb(51, 51, 51);">解决这个问题的思路与常见的跳表算法相似，如下图要搜索跳表，从最高层开始，沿着具有最长 “跳过” 的边向右移动。如果发现当前节点的值大于要搜索的值 - 我们知道已经超过了目标，因此我们会在下一级中向前一个节点。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1729395383463-f8370948-9d00-4bb4-ba57-014a6560db39.png)

![](https://cdn.nlark.com/yuque/0/2024/gif/2424104/1729735738973-3dd80195-f755-4edb-a5ec-c9ce05ba45bb.gif)

<font style="color:rgb(51, 51, 51);">HNSW 继承了相同的分层格式，最高层具有更长的边缘（用于快速搜索），而较低层具有较短的边缘（用于准确搜索）。</font>

<font style="color:rgb(51, 51, 51);">具体来说，可以将图分为多层，每一层都是一个小世界，图中的节点都是相互连接的。而且每一层的节点都会连接到上一层的节点，当需要搜索的时候，就可以从第一层开始，因为第一层的节点之间距离很长，可以减少搜索的时间，然后再逐层向下搜索，又因为最下层相似节点之间相互关联，所以可以保证搜索的质量，能够找到最相似的向量。</font>

<font style="color:rgb(51, 51, 51);">HNSW 算法是一种经典的空间换时间的算法，它的搜索质量和搜索速度都比较高，但是它的内存开销也比较大，因为不仅需要将所有的向量都存储在内存中。还需要维护一个图的结构，也同样需要存储。所以这类算法需要根据实际的场景来选择。</font>

**通过一个简单的例子来说明 HNSW 的工作原理。**

<font style="color:rgb(51, 51, 51);">假设我们有以下五个二维向量：</font>

+ $ A = (1, 2) $
+ $ B = (2, 3) $
+ $ C = (3, 4) $
+ $ D = (8, 9) $
+ $ E = (9, 10) $

<font style="color:rgb(51, 51, 51);">我们要使用 HNSW 来找到与查询向量 </font>$ Q = (2, 2) $<font style="color:rgb(51, 51, 51);"> 最相似的向量。</font>

**<font style="color:rgb(51, 51, 51);">构建阶段</font>**

1. **初始化**<font style="color:rgb(51, 51, 51);">：</font>
    - <font style="color:rgb(51, 51, 51);">首先，我们为每个向量随机选择其在 HNSW 中的层数。例如，假设：</font>
        * $ A $<font style="color:rgb(51, 51, 51);"> 在第2层和第1层</font>
        * $ B $<font style="color:rgb(51, 51, 51);"> 在第1层</font>
        * $ C $<font style="color:rgb(51, 51, 51);"> 在第1层</font>
        * $ D $<font style="color:rgb(51, 51, 51);"> 在第1层</font>
        * $ E $<font style="color:rgb(51, 51, 51);"> 在第1层</font>
2. **构建图**<font style="color:rgb(51, 51, 51);">：</font>
    - <font style="color:rgb(51, 51, 51);">从最高层开始构建，</font>$ A $<font style="color:rgb(51, 51, 51);"> 是第2层唯一的节点。</font>
    - <font style="color:rgb(51, 51, 51);">在第1层，所有节点都存在。我们通过计算欧氏距离来连接节点，确保每个节点只连接到几个最近的邻居。</font>

**<font style="color:rgb(51, 51, 51);">搜索阶段</font>**

1. **从最高层开始搜索**<font style="color:rgb(51, 51, 51);">：</font>
    - <font style="color:rgb(51, 51, 51);">从第2层的 </font>$ A $<font style="color:rgb(51, 51, 51);"> 开始，计算 </font>$ Q $<font style="color:rgb(51, 51, 51);"> 与 </font>$ A $<font style="color:rgb(51, 51, 51);"> 的距离：</font>$ d(Q, A) = \sqrt{(2-1)^2 + (2-2)^2} = 1 $<font style="color:rgb(51, 51, 51);">。</font>
    - <font style="color:rgb(51, 51, 51);">因为 </font>$ A $<font style="color:rgb(51, 51, 51);"> 是唯一的节点，直接进入下一层。</font>
2. **在第1层搜索**<font style="color:rgb(51, 51, 51);">：</font>
    - <font style="color:rgb(51, 51, 51);">从 </font>$ A $<font style="color:rgb(51, 51, 51);"> 开始，计算 </font>$ Q $<font style="color:rgb(51, 51, 51);"> 与 </font>$ B $<font style="color:rgb(51, 51, 51);">、</font>$ C $<font style="color:rgb(51, 51, 51);">、</font>$ D $<font style="color:rgb(51, 51, 51);">、</font>$ E $<font style="color:rgb(51, 51, 51);"> 的距离：</font>
        * $ d(Q, B) = \sqrt{(2-2)^2 + (2-3)^2} = 1 $
        * $ d(Q, C) = \sqrt{(2-3)^2 + (2-4)^2} = \sqrt{5} $
        * $ d(Q, D) = \sqrt{(2-8)^2 + (2-9)^2} = \sqrt{85} $
        * $ d(Q, E) = \sqrt{(2-9)^2 + (2-10)^2} = \sqrt{113} $
    - $ B $<font style="color:rgb(51, 51, 51);"> 是最近的邻居。</font>
3. **返回结果**<font style="color:rgb(51, 51, 51);">：</font>
    - <font style="color:rgb(51, 51, 51);">在第2层找到的最近邻是 </font>$ A $<font style="color:rgb(51, 51, 51);">，第1层找到的最近邻是 </font>$ B $<font style="color:rgb(51, 51, 51);">，即与查询向量 </font>$ Q $<font style="color:rgb(51, 51, 51);"> 最相似的向量。</font>

**<font style="color:rgb(51, 51, 51);">总结</font>**

<font style="color:rgb(51, 51, 51);">通过这种层次化的搜索过程，HNSW 能够快速缩小搜索范围，在大规模数据中高效找到近似最近邻。</font>

# 基于 Embedding 的问答助手和意图匹配
## <font style="color:rgb(28, 30, 33);">Embedding models(嵌入模型)</font>
<font style="color:rgb(28, 30, 33);">Embeddings类是一个专为与文本嵌入模型进行交互而设计的类。有许多嵌入模型提供商（如OpenAI、Cohere、Hugging Face等）- 这个类旨在为它们提供一个标准接口。</font>

<font style="color:rgb(28, 30, 33);">Embeddings类会为文本创建一个向量表示。这很有用，因为这意味着我们可以在向量空间中思考文本，并做一些类似语义搜索的事情，比如在向量空间中寻找最相似的文本片段。</font>

<font style="color:rgb(28, 30, 33);">LangChain中的基本Embeddings类提供了两种方法：一个用于嵌入文档，另一个用于嵌入查询。前者</font>`<font style="color:rgb(28, 30, 33);">.embed_documents</font>`<font style="color:rgb(28, 30, 33);">接受多个文本作为输入，而后者</font>`<font style="color:rgb(28, 30, 33);">.embed_query</font>`<font style="color:rgb(28, 30, 33);">接受单个文本。之所以将它们作为两个单独的方法，是因为一些嵌入提供商对文档（要搜索的文档）和查询（搜索查询本身）有不同的嵌入方法。</font>

`<font style="color:rgb(28, 30, 33);">.embed_query</font>`<font style="color:rgb(28, 30, 33);">将返回一个浮点数列表，而</font>`<font style="color:rgb(28, 30, 33);">.embed_documents</font>`<font style="color:rgb(28, 30, 33);">将返回一个浮点数列表的列表。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1722780976370-0672d8a6-e5e4-4a60-b303-11913514d81d.png)

## 设置
+ OpenAI

<font style="color:rgb(28, 30, 33);">首先，我们需要安装OpenAI合作伙伴包：</font>

```bash
pip install langchain-openai
```

<font style="color:rgb(28, 30, 33);">访问API需要一个API密钥，您可以通过创建帐户并转到</font>[<font style="color:rgb(28, 30, 33);">这里</font>](https://platform.openai.com/account/api-keys)<font style="color:rgb(28, 30, 33);">来获取它。一旦我们有了密钥，我们希望通过运行以下命令将其设置为环境变量：</font>

```bash
export OPENAI_API_KEY="..."
```

<font style="color:rgb(28, 30, 33);">如果您不想设置环境变量，可以在初始化OpenAI LLM类时通过</font>`<font style="color:rgb(28, 30, 33);">api_key</font>`<font style="color:rgb(28, 30, 33);">命名参数直接传递密钥：</font>

```python
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(api_key="...")
```

<font style="color:rgb(28, 30, 33);">否则，您可以不带任何参数初始化：</font>

```python
from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()
```



## 嵌入文本列表(embed_documents)
<font style="color:rgb(28, 30, 33);">使用</font>`<font style="color:rgb(28, 30, 33);">.embed_documents</font>`<font style="color:rgb(28, 30, 33);">来嵌入一个字符串列表，恢复一个嵌入列表：</font>

```python
#示例：embed_documents.py
embeddings = embeddings_model.embed_documents(
    [
        "嗨！",
        "哦，你好！",
        "你叫什么名字？",
        "我的朋友们叫我World",
        "Hello World！"
    ]
)
len(embeddings), len(embeddings[0])
```

```plain
(5, 1536)
```

## 嵌入单个查询(embed_query)
<font style="color:rgb(28, 30, 33);">使用</font>`<font style="color:rgb(28, 30, 33);">.embed_query</font>`<font style="color:rgb(28, 30, 33);">来嵌入单个文本片段（例如，用于与其他嵌入的文本片段进行比较）。</font>

```python
#示例：embed_query.py
embedded_query = embeddings_model.embed_query("对话中提到的名字是什么？")
embedded_query[:5]
```

```plain
[0.003292066277936101, -0.009463472291827202, 0.03418809175491333, -0.0011715596774592996, -0.015508134849369526]
```

---

## 问答助手
```python
#示例：embed_search.py
from openai import OpenAI
#pip install numpy
from numpy import dot
from numpy.linalg import norm

client = OpenAI()


# 定义调用 Embedding API 的函数
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


# 计算余弦相似度，参考文章：https://blog.csdn.net/Hyman_Qiu/article/details/137743190
#定义一个函数 cosine_similarity，该函数接受两个向量 vec1 和 vec2 作为输入。
def cosine_similarity(vec1, vec2):
    #计算并返回两个向量之间的余弦相似度，公式为：两个向量的点积除以它们范数的乘积。
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 实现文本搜索功能
# 定义一个函数 search_documents，该函数接受一个查询字符串 query 和一个文档列表 documents 作为输入。
def search_documents(query, documents):
    # 调用 get_embedding 函数生成查询字符串的嵌入向量 query_embedding。
    query_embedding = get_embedding(query)
    # 对每个文档调用 get_embedding 函数生成文档的嵌入向量，存储在 document_embeddings 列表中。
    document_embeddings = [get_embedding(doc) for doc in documents]
    # 计算查询嵌入向量与每个文档嵌入向量之间的余弦相似度，存储在 similarities 列表中
    similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
    # 找到相似度最高的文档的索引 most_similar_index。
    most_similar_index = similarities.index(max(similarities))
    # 返回相似度最高的文档和相似度得分。
    return documents[most_similar_index], max(similarities)


# 测试文本搜索功能
if __name__ == "__main__":
    documents = [
        "OpenAI的ChatGPT是一个强大的语言模型。",
        "天空是蓝色的,阳光灿烂。",
        "人工智能正在改变世界。",
        "Python是一种流行的编程语言。"
    ]
    #".././resource/knowledge.txt" 文件内容，转换为上述documents
    #documents = [line.strip() for line in open(".././resource/knowledge.txt", "r", encoding="utf-8")]

    query = "天空是什么颜色的？"

    most_similar_document, similarity_score = search_documents(query, documents)
    print(f"最相似的文档: {most_similar_document}")
    print(f"相似性得分: {similarity_score}")

```



```plain
最相似的文档: 天空是蓝色的,阳光灿烂。
相似性得分: 0.9166142096488017
```

## 意图匹配
```python
#示例：embed_intention_recognition.py
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# 数据导入
loader = TextLoader(".././resource/qa.txt", encoding="UTF-8")
docs = loader.load()
# 数据切分
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# 创建embedding
embeddings = OpenAIEmbeddings()
# 通过向量数据库存储
vector = FAISS.from_documents(documents, embeddings)
# 查询检索
# 创建 prompt
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答以下问题：:
<context>
{context}
</context>
Question: {input}""")
# 创建模型
llm = ChatOpenAI()
# 创建 document 的chain， 查询
document_chain = create_stuff_documents_chain(llm, prompt)
from langchain.chains import create_retrieval_chain

# # 创建搜索chain 返回值为 VectorStoreRetriever
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# # 执行请求
response = retrieval_chain.invoke({"input": "天气"})
print(response["answer"])

```



```python
抱歉，我无法回答关于天气的问题。如果您需要查询天气信息，请使用天气应用程序或者访问天气网站。我可以帮您订餐、播放音乐、设置闹钟或者讲笑话。有什么其他问题我可以帮助您解决吗？
```



# 基于 LangChain 构建向量存储和查询：Chroma, Weaviate, Qdrant,  Milvus
## Vector stores(<font style="color:rgb(28, 30, 33);">向量存储)</font>
<font style="color:rgb(28, 30, 33);">存储和搜索非结构化数据的最常见方法之一是将其嵌入并存储生成的嵌入向量， 然后在查询时将非结构化查询嵌入并检索与嵌入查询“最相似”的嵌入向量。 向量存储会处理存储嵌入数据并为您执行向量搜索。 可以通过以下方式将向量存储转换为检索器接口：</font>

<font style="color:rgb(28, 30, 33);"></font>

<font style="color:rgb(28, 30, 33);">Retrievers(检索器)是一个接口，根据非结构化查询返回文档。 它比向量存储更通用。 检索器不需要能够存储文档，只需要能够返回（或检索）它们。 检索器可以从向量存储器创建，但也足够广泛，包括Wikipedia搜索和Amazon Kendra。 检索器接受字符串查询作为输入，并返回文档列表作为输出。</font>

```python
vectorstore = MyVectorStore()
retriever = vectorstore.as_retriever()
```



## <font style="color:rgb(28, 30, 33);">Chroma</font>
<font style="color:rgb(28, 30, 33);">官方文档：</font>[https://docs.trychroma.com/](https://docs.trychroma.com/)

[<font style="color:rgb(28, 30, 33);">Chroma</font>](https://docs.trychroma.com/getting-started)<font style="color:rgb(28, 30, 33);"> ( /'kromə/ n. （色彩的）浓度，色度 )是一个以人工智能为基础的开源向量数据库，专注于开发者的生产力和幸福感。Chroma 使用 Apache 2.0 许可证。 使用以下命令安装 Chroma：</font>

```plain
pip install langchain-chroma
```

<font style="color:rgb(28, 30, 33);">Chroma 可以以多种模式运行。以下是每种模式的示例，均与 LangChain 集成：</font>

+ `<font style="color:rgb(28, 30, 33);">in-memory</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 在 Python 脚本或 Jupyter 笔记本中</font>
+ `<font style="color:rgb(28, 30, 33);">in-memory with persistance</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 在脚本或笔记本中保存/加载到磁盘</font>
+ `<font style="color:rgb(28, 30, 33);">in a docker container</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">- 作为在本地机器或云中运行的服务器</font>

<font style="color:rgb(28, 30, 33);">与任何其他数据库一样，您可以进行以下操作：</font>

+ `<font style="color:rgb(28, 30, 33);">.add</font>`
+ `<font style="color:rgb(28, 30, 33);">.get</font>`
+ `<font style="color:rgb(28, 30, 33);">.update</font>`
+ `<font style="color:rgb(28, 30, 33);">.upsert</font>`
+ `<font style="color:rgb(28, 30, 33);">.delete</font>`
+ `<font style="color:rgb(28, 30, 33);">.peek</font>`
+ <font style="color:rgb(28, 30, 33);">而</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">.query</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">则运行相似性搜索。</font>

<font style="color:rgb(28, 30, 33);">查看完整文档，请访问</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">docs</font>](https://docs.trychroma.com/reference/Collection)<font style="color:rgb(28, 30, 33);">。要直接访问这些方法，可以使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">._collection.method()</font>`<font style="color:rgb(28, 30, 33);">。</font>

### <font style="color:rgb(28, 30, 33);">基本示例</font>
<font style="color:rgb(28, 30, 33);">在这个基本示例中，我们获取《乔布斯演讲稿》，将其分割成片段，使用开源嵌入模型进行嵌入，加载到 Chroma 中，然后进行查询。</font>

```python
#示例：chroma_base.py
# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# 创建开源嵌入函数
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 将其加载到 Chroma 中
db = Chroma.from_documents(docs, embedding_function)
# 进行查询
query = "Pixar公司是做什么的?"
docs = db.similarity_search(query)
# 打印结果
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

### <font style="color:rgb(28, 30, 33);">基本示例（包括保存到磁盘）</font>
<font style="color:rgb(28, 30, 33);">在上一个示例的基础上，如果您想要保存到磁盘，只需初始化 Chroma 客户端并传递要保存数据的目录。</font>

`<font style="color:rgb(28, 30, 33);">注意</font>`<font style="color:rgb(28, 30, 33);">：Chroma 尽最大努力自动将数据保存到磁盘，但多个内存客户端可能会相互干扰。最佳做法是，任何给定时间只运行一个客户端。</font>

```python
#示例：chroma_disk.py

# 保存到磁盘
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
docs = db2.similarity_search(query)
# 从磁盘加载
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

### <font style="color:rgb(28, 30, 33);">将 Chroma 客户端传递给 Langchain</font>
<font style="color:rgb(28, 30, 33);">您还可以创建一个 Chroma 客户端并将其传递给 LangChain。如果您希望更轻松地访问底层数据库，这将特别有用。</font>

<font style="color:rgb(28, 30, 33);">您还可以指定要让 LangChain 使用的集合名称。</font>

```python
#示例：chroma_client.py
import chromadb
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection_name")
collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])
langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embedding_function,
)
print("在集合中有", langchain_chroma._collection.count(), "个文档")
```

```plain
Insert of existing embedding ID: 1
Insert of existing embedding ID: 2
Insert of existing embedding ID: 3
Add of existing embedding ID: 1
Add of existing embedding ID: 2
Add of existing embedding ID: 3
在集合中有 3 个项目

```

### <font style="color:rgb(28, 30, 33);">更新和删除</font>
<font style="color:rgb(28, 30, 33);">在构建真实应用程序时，您不仅希望添加数据，还希望更新和删除数据。</font>

<font style="color:rgb(28, 30, 33);">Chroma 要求用户提供</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">ids</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">来简化这里的簿记工作。</font>`<font style="color:rgb(28, 30, 33);">ids</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">可以是文件名，也可以是类似</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">filename_paragraphNumber</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的组合哈希值。</font>

<font style="color:rgb(28, 30, 33);">Chroma 支持所有这些操作，尽管有些操作仍在通过 LangChain 接口进行整合。额外的工作流改进将很快添加。</font>

<font style="color:rgb(28, 30, 33);">以下是一个基本示例，展示如何执行各种操作：</font>

```python
#示例：chroma_update.py
# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# 创建开源嵌入函数
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query = "Pixar公司是做什么的?"
# 创建简单的 ids
ids = [str(i) for i in range(1, len(docs) + 1)]
# 添加数据
example_db = Chroma.from_documents(docs, embedding_function, ids=ids)
docs = example_db.similarity_search(query)
# 更新文档的元数据
docs[0].metadata = {
    "source": "../../resource/knowledge.txt",
    "new_value": "hello world",
}
print("更新前内容：")
print(example_db._collection.get(ids=[ids[0]]))
example_db.update_document(ids[0], docs[0])
print("更新后内容：")
print(example_db._collection.get(ids=[ids[0]]))
# 删除最后一个文档
print("删除前计数", example_db._collection.count())
print(example_db._collection.get(ids=[ids[-1]]))
example_db._collection.delete(ids=[ids[-1]])
print("删除后计数", example_db._collection.count())
print(example_db._collection.get(ids=[ids[-1]]))
```

```markdown
更新前内容：
{'ids': ['1'], 'embeddings': None, 'metadatas': [{'source': '../../resource/knowledge.txt'}], 'documents': ["\ufeffI am honored to be with you today at your commencement from one of the finest universities in the world. I never graduated from college. Truth be told, this is the closest I've ever gotten to a college graduation. Today I want to tell you three stories from my life. That's it. No big deal. Just three stories.\n我今天很荣幸能和你们一起参加毕业典礼，斯坦福大学是世界上最好的大学之一。我从来没有从大学中毕业。说实话,今天也许是在我的生命中离大学毕业最近的一天了。今天我想向你们讲述我生活中的三个故事。不是什么大不了的事情,只是三个故事而已。\n\nThe first story is about connecting the dots.\n第一个故事是关于如何把生命中的点点滴滴串连起来。\n\nI dropped out of Reed College after the first 6 months, but then stayed around as a drop-in for another 18 months or so before I really quit. So why did I drop out?\n我在Reed大学读了六个月之后就退学了，但是在十八个月以后——我真正的作出退学决定之前，我还经常去学校。我为什么要退学呢？"], 'uris': None, 'data': None, 'included': ['metadatas', 'documents']}
更新后内容：
{'ids': ['1'], 'embeddings': None, 'metadatas': [{'new_value': 'hello world', 'source': '../../resource/knowledge.txt'}], 'documents': ["During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.\n在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。"], 'uris': None, 'data': None, 'included': ['metadatas', 'documents']}
删除前计数 16
{'ids': ['16'], 'embeddings': None, 'metadatas': [{'source': '../../resource/knowledge.txt'}], 'documents': ['Stewart and his team put out several issues of The Whole Earth Catalog, and then when it had run its course, they put out a final issue. It was the mid-1970s, and I was your age. On the back cover of their final issue was a photograph of an early morning country road, the kind you might find yourself hitchhiking on if you were so adventurous. Beneath it were the words: "Stay Hungry. Stay Foolish." It was their farewell message as they signed off. Stay Hungry. Stay Foolish. And I have always wished that for myself. And now, as you graduate to begin anew, I wish that for you.\nStewart和他的伙伴出版了几期的“整个地球的目录”，当它完成了自己使命的时候，他们做出了最后一期的目录。那是在七十年代的中期，你们的时代。在最后一期的封底上是清晨乡村公路的照片（如果你有冒险精神的话，你可以自己找到这条路的），在照片之下有这样一段话：“求知若饥，虚心若愚。”这是他们停止了发刊的告别语。“求知若饥，虚心若愚。”我总是希望自己能够那样，现在，在你们即将毕业，开始新的旅程的时候，我也希望你们能这样：\n\nStay Hungry. Stay Foolish.\n求知若饥，虚心若愚。\n\nThank you all very much.\n非常感谢你们。'], 'uris': None, 'data': None, 'included': ['metadatas', 'documents']}
删除后计数 15
{'ids': [], 'embeddings': None, 'metadatas': [], 'documents': [], 'uris': None, 'data': None, 'included': ['metadatas', 'documents']}
```



### <font style="color:rgb(28, 30, 33);">使用 OpenAI Embeddings</font>
<font style="color:rgb(28, 30, 33);">许多人喜欢使用 OpenAIEmbeddings，以下是如何设置它。</font>

```python
#示例：chroma_openai.py
from langchain_openai import OpenAIEmbeddings
# pip install langchain-chroma
from langchain_chroma import Chroma
import chromadb
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

embeddings = OpenAIEmbeddings()
persistent_client = chromadb.PersistentClient()
new_client = chromadb.EphemeralClient()
# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

openai_lc_client = Chroma.from_documents(
    docs, embeddings, client=new_client, collection_name="openai_collection"
)
query = "Pixar公司是做什么的?"
docs = openai_lc_client.similarity_search(query)
print(docs[0].page_content)

```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

---

### <font style="color:rgb(28, 30, 33);">其他功能</font>
#### <font style="color:rgb(28, 30, 33);">带分数的相似性搜索</font>
<font style="color:rgb(28, 30, 33);">返回的距离分数是余弦距离。因此，分数越低越好。</font>

```python
#示例：chroma_other.py
docs = db.similarity_search_with_score(query)
```

```python
docs[0]
#0.8513926863670349
```

```plain
(Document(metadata={'source': '../../resource/knowledge.txt'}, page_content="During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.\n在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。"), 0.8513926863670349)
```

#### <font style="color:rgb(28, 30, 33);">检索器选项</font>
<font style="color:rgb(28, 30, 33);">本节介绍如何在检索器中使用 Chroma 的不同选项。</font>

**<font style="color:rgb(28, 30, 33);">MMR</font>**

<font style="color:rgb(28, 30, 33);">除了在检索器对象中使用相似性搜索外，还可以使用 MMR（Maximal Marginal Relevance，最大边际相关性）是一种信息检索和文本摘要技术，用于在选择文档或文本片段时平衡相关性和多样性。其主要目的是在检索结果中既包含与查询高度相关的内容，又避免结果之间的高度冗余。</font>

**<font style="color:rgb(28, 30, 33);">MMR的作用</font>**

1. <font style="color:rgb(28, 30, 33);">提高结果的多样性：通过引入多样性，MMR可以避免检索结果中出现重复信息，从而提供更全面的答案。</font>
2. <font style="color:rgb(28, 30, 33);">平衡相关性和新颖性：MMR在选择结果时，既考虑与查询的相关性，也考虑新信息的引入，以确保结果的多样性和覆盖面。</font>
3. <font style="color:rgb(28, 30, 33);">减少冗余：通过避免选择与已选结果高度相似的文档，MMR可以减少冗余，提高信息的利用效率。</font>

**<font style="color:rgb(28, 30, 33);">在检索器对象中使用MMR</font>**

<font style="color:rgb(28, 30, 33);">在使用向量检索器（如FAISS）时，通常通过相似性搜索来查找与查询最相关的文档。然而，这种方法有时可能会返回许多相似的结果，导致信息冗余。MMR可以在这种情况下发挥作用，通过以下步骤实现：</font>

1. <font style="color:rgb(28, 30, 33);">计算相关性：首先，计算每个候选文档与查询的相似性得分。</font>
2. <font style="color:rgb(28, 30, 33);">计算多样性：然后，计算每个候选文档与已选文档集合的相似性得分。</font>
3. <font style="color:rgb(28, 30, 33);">选择文档：在每一步选择一个文档，使得该文档在相关性和多样性之间达到最佳平衡。</font>

```python
#示例：chroma_other.py
retriever = db.as_retriever(search_type="mmr")
```

```python
retriever.invoke(query)[0]
```

```plain
page_content='During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。' metadata={'source': '../../resource/knowledge.txt'}
```



## Weaviate
<font style="color:rgb(28, 30, 33);">如何使用 </font>`<font style="color:rgb(28, 30, 33);">langchain-weaviate</font>`<font style="color:rgb(28, 30, 33);"> 包在 LangChain 中开始使用 Weaviate 向量存储。</font>

[Weaviate](https://weaviate.io/) 是一个开源的向量数据库。它允许您存储来自您喜爱的机器学习模型的数据对象和向量嵌入，并能够无缝地扩展到数十亿个数据对象。

官方文档：[https://weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)

<font style="color:rgb(28, 30, 33);">要使用此集成，您需要运行一个 Weaviate 数据库实例。</font>

### 最低版本
<font style="color:rgb(28, 30, 33);">此模块需要 Weaviate</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">1.23.7</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">或更高版本。但是，我们建议您使用最新版本的 Weaviate。</font>

### 连接到 Weaviate
<font style="color:rgb(28, 30, 33);">在本文中，我们假设您在</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">http://localhost:8080</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">上运行了一个本地的 Weaviate 实例，并且端口 50051 用于</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">gRPC 通信</font>](https://weaviate.io/blog/grpc-performance-improvements)<font style="color:rgb(28, 30, 33);">。因此，我们将使用以下代码连接到 Weaviate：</font>

```python
weaviate_client = weaviate.connect_to_local()
```

#### 其他部署选项
<font style="color:rgb(28, 30, 33);">Weaviate 可以以许多不同的方式进行</font>[<font style="color:rgb(28, 30, 33);">部署</font>](https://weaviate.io/developers/weaviate/starter-guides/which-weaviate)<font style="color:rgb(28, 30, 33);">，例如使用</font>[<font style="color:rgb(28, 30, 33);">Weaviate Cloud Services (WCS)</font>](https://console.weaviate.cloud/)<font style="color:rgb(28, 30, 33);">、</font>[<font style="color:rgb(28, 30, 33);">Docker</font>](https://weaviate.io/developers/weaviate/installation/docker-compose)<font style="color:rgb(28, 30, 33);">或</font>[<font style="color:rgb(28, 30, 33);">Kubernetes</font>](https://weaviate.io/developers/weaviate/installation/kubernetes)<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">如果您的 Weaviate 实例以其他方式部署，可以在</font>[<font style="color:rgb(28, 30, 33);">此处阅读更多信息</font>](https://weaviate.io/developers/weaviate/client-libraries/python#instantiate-a-client)<font style="color:rgb(28, 30, 33);">关于连接到 Weaviate 的不同方式。您可以使用不同的</font>[<font style="color:rgb(28, 30, 33);">辅助函数</font>](https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-helper-functions)<font style="color:rgb(28, 30, 33);">，或者</font>[<font style="color:rgb(28, 30, 33);">创建一个自定义实例</font>](https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-explicit-connection)<font style="color:rgb(28, 30, 33);">。</font>

请注意，您需要一个 `v4` 客户端 API，它将创建一个 `weaviate.WeaviateClient` 对象。

#### 认证
<font style="color:rgb(28, 30, 33);">一些 Weaviate 实例，例如在 WCS 上运行的实例，启用了认证，例如 API 密钥和/或用户名+密码认证。</font>

<font style="color:rgb(28, 30, 33);">阅读</font>[<font style="color:rgb(28, 30, 33);">客户端认证指南</font>](https://weaviate.io/developers/weaviate/client-libraries/python#authentication)<font style="color:rgb(28, 30, 33);">以获取更多信息，以及</font>[<font style="color:rgb(28, 30, 33);">深入的认证配置页面</font>](https://weaviate.io/developers/weaviate/configuration/authentication)<font style="color:rgb(28, 30, 33);">。</font>

### 安装
```python
# 安装包
# %pip install -Uqq langchain-weaviate
# %pip install openai tiktoken langchain
```

### 环境设置
<font style="color:rgb(28, 30, 33);">本文使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">OpenAIEmbeddings</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">通过 OpenAI API。我们建议获取一个 OpenAI API 密钥，并将其作为名为</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">OPENAI_API_KEY</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的环境变量导出。</font>

<font style="color:rgb(28, 30, 33);">完成后，您的 OpenAI API 密钥将被自动读取。如果您对环境变量不熟悉，可以在</font>[<font style="color:rgb(28, 30, 33);">此处</font>](https://docs.python.org/3/library/os.html#os.environ)<font style="color:rgb(28, 30, 33);">或</font>[<font style="color:rgb(28, 30, 33);">此指南</font>](https://www.twilio.com/en-us/blog/environment-variables-python)<font style="color:rgb(28, 30, 33);">中阅读更多关于它们的信息。</font>

<font style="color:rgb(28, 30, 33);">配置Weaviate的WCD_DEMO_URL和WCD_DEMO_RO_KEY</font>

```powershell
setx WCD_DEMO_URL ""
setx WCD_DEMO_RO_KEY ""
```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724049541827-43f841c0-1312-41fa-997c-aaf5d12c74ca.png)

### 用法
### 通过相似性查找对象
<font style="color:rgb(28, 30, 33);">以下是一个示例，演示如何通过查询查找与之相似的对象，从数据导入到查询 Weaviate 实例。</font>

#### 步骤 1：数据导入
<font style="color:rgb(28, 30, 33);">首先，我们将创建要添加到</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">Weaviate</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的数据，方法是加载并分块长文本文件的内容。</font>

```python
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
```

<font style="color:rgb(28, 30, 33);">现在，我们可以导入数据。要这样做，连接到 Weaviate 实例，并使用生成的 </font>`<font style="color:rgb(28, 30, 33);">weaviate_client</font>`<font style="color:rgb(28, 30, 33);"> 对象。例如，我们可以将文档导入如下所示：</font>

```python
#示例：weaviate_client.py
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': openai_api_key}  # Replace with your OpenAI API key
)
db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)
```

#### 第二步：执行搜索
<font style="color:rgb(28, 30, 33);">现在我们可以执行相似度搜索。这将返回与查询文本最相似的文档，基于存储在 Weaviate 中的嵌入和从查询文本生成的等效嵌入。</font>

```python
#示例：weaviate_search.py
# pip install -Uqq langchain-weaviate
# pip install openai tiktoken langchain
import os
import weaviate
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth

embeddings = OpenAIEmbeddings()
# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Best practice: store your credentials in environment variables
wcd_url = os.environ["WCD_DEMO_URL"]
wcd_api_key = os.environ["WCD_DEMO_RO_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': openai_api_key}  # Replace with your OpenAI API key
)
db = WeaviateVectorStore.from_documents(docs, embeddings, client=weaviate_client)
query = "Pixar公司是做什么的?"
docs = db.similarity_search(query)
print(docs[0].page_content)
weaviate_client.close()
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

#### 量化结果相似性
<font style="color:rgb(28, 30, 33);">您可以选择检索相关性“分数”。这是一个相对分数，表示特定搜索结果在搜索结果池中的好坏程度。</font>

<font style="color:rgb(28, 30, 33);">请注意，这是相对分数，意味着不应用于确定相关性的阈值。但是，它可用于比较整个搜索结果集中不同搜索结果的相关性。</font>

```python
#示例：weaviate_similarity.py
query = "Pixar公司是做什么的?"
docs = db.similarity_search_with_score(query, k=5)
for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")
```

```plain
0.700 : During the next five years, I started a company named NeXT, another company named Pixar, and fell in...
0.337 : I was lucky – I found what I loved to do early in life. Woz and I started Apple in my parents garage...
0.271 : I really didn't know what to do for a few months. I felt that I had let the previous generation of e...
0.256 : I'm pretty sure none of this would have happened if I hadn't been fired from Apple. It was awful tas...
0.191 : Stewart and his team put out several issues of The Whole Earth Catalog, and then when it had run its...
```

### 
## <font style="color:rgb(28, 30, 33);">Qdrant</font>
### Qdrant介绍
[<font style="color:rgb(28, 30, 33);">Qdrant</font>](https://qdrant.tech/documentation/)<font style="color:rgb(28, 30, 33);">（读作：quadrant  /'kwɑdrənt/  n. 象限；象限仪；四分之一圆）是一个向量相似度搜索引擎。它提供了一个生产就绪的服务，具有方便的 API 来存储、搜索和管理点 - 带有附加载荷的向量。</font>`<font style="color:rgb(28, 30, 33);">Qdrant</font>`<font style="color:rgb(28, 30, 33);">专门支持扩展过滤功能，使其对各种神经网络或基于语义的匹配、分面搜索和其他应用非常有用。</font>

<font style="color:rgb(28, 30, 33);">官方文档：</font>[https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)

<font style="color:rgb(28, 30, 33);">以下展示了如何使用与</font>`<font style="color:rgb(28, 30, 33);">Qdrant</font>`<font style="color:rgb(28, 30, 33);">向量数据库相关的功能。</font>

<font style="color:rgb(28, 30, 33);">有各种运行</font>`<font style="color:rgb(28, 30, 33);">Qdrant</font>`<font style="color:rgb(28, 30, 33);">的模式，取决于所选择的模式，会有一些细微的差异。选项包括：</font>

+ <font style="color:rgb(28, 30, 33);">本地模式，无需服务器</font>
+ <font style="color:rgb(28, 30, 33);">Qdrant 云</font>

<font style="color:rgb(28, 30, 33);">请参阅</font>[<font style="color:rgb(28, 30, 33);">安装说明</font>](https://qdrant.tech/documentation/install/)<font style="color:rgb(28, 30, 33);">。</font>

```python
%pip install --upgrade --quiet  langchain-qdrant langchain-openai langchain
```

```python
#示例：qdrant_local.py
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_text_splitters import CharacterTextSplitter
```

```python
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
```

### <font style="color:rgb(28, 30, 33);">本地模式</font>
<font style="color:rgb(28, 30, 33);">Python 客户端允许您在本地模式下运行相同的代码，而无需运行 Qdrant 服务器。这对于测试和调试或者如果您计划仅存储少量向量非常有用。嵌入可能完全保存在内存中或者持久化在磁盘上。</font>

#### <font style="color:rgb(28, 30, 33);">内存中</font>
<font style="color:rgb(28, 30, 33);">对于一些测试场景和快速实验，您可能更喜欢仅将所有数据保存在内存中，因此当客户端被销毁时数据会丢失 - 通常在脚本/笔记本的末尾。</font>

```python
#示例：qdrant_local.py
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # 本地模式，仅内存存储
    collection_name="my_documents",
)
```

#### <font style="color:rgb(28, 30, 33);">磁盘存储</font>
<font style="color:rgb(28, 30, 33);">在不使用 Qdrant 服务器的本地模式下，还可以将您的向量存储在磁盘上，以便它们在运行之间持久化。</font>

```python
#示例：qdrant_disk.py
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    path="/tmp/local_qdrant",
    collection_name="my_documents",
)
```

### <font style="color:rgb(28, 30, 33);">相似度搜索</font>
<font style="color:rgb(28, 30, 33);">使用 Qdrant 向量存储的最简单场景是执行相似度搜索。在幕后，我们的查询将使用</font>`<font style="color:rgb(28, 30, 33);">embedding_function</font>`<font style="color:rgb(28, 30, 33);">对查询进行编码，并用于在 Qdrant 集合中查找相似的文档。</font>

```python
query = "Pixar公司是做什么的?"
found_docs = qdrant.similarity_search(query)
```

```python
print(found_docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

 

### <font style="color:rgb(28, 30, 33);">Qdrant 云</font>
<font style="color:rgb(28, 30, 33);">如果您不想忙于管理基础设施，可以选择在 </font>[<font style="color:rgb(28, 30, 33);">Qdrant 云</font>](https://cloud.qdrant.io/)<font style="color:rgb(28, 30, 33);">上设置一个完全托管的 Qdrant 集群。包括一个永久免费的 1GB 集群供试用。使用托管版本与使用 Qdrant 的主要区别在于，您需要提供 API 密钥以防止公开访问您的部署。该值也可以设置在 </font>`<font style="color:rgb(28, 30, 33);">QDRANT_API_KEY</font>`<font style="color:rgb(28, 30, 33);"> 环境变量中。</font>

```python
url = "<---qdrant cloud cluster url here --->"
api_key = "<---api key here--->"
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="my_documents",
)
```



## Milvus
### [<font style="color:rgb(28, 30, 33);">Milvus</font>](https://milvus.io/docs/overview.md)<font style="color:rgb(28, 30, 33);">介绍</font>
[<font style="color:rgb(28, 30, 33);">Milvus</font>](https://milvus.io/docs/overview.md)<font style="color:rgb(28, 30, 33);"> 是一个数据库，用于存储、索引和管理由深度神经网络和其他机器学习（ML）模型生成的大规模嵌入向量。</font>

<font style="color:rgb(28, 30, 33);">官方文档：</font>[https://milvus.io/docs/overview.md](https://milvus.io/docs/overview.md)

<font style="color:rgb(28, 30, 33);">下面讲解如何使用与 Milvus 向量数据库相关的功能。</font>

<font style="color:rgb(28, 30, 33);">配置环境变量MILVUS_API_URL和MILVUS_API_KEY</font>

```powershell
setx MILVUS_API_URL ""
setx MILVUS_API_KEY ""
```

zilliz：[https://cloud.zilliz.com/](https://cloud.zilliz.com/)

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724049217804-83fe8bbd-04c9-4fb6-9793-89e9befba746.png)

<font style="color:rgb(28, 30, 33);">要运行，您应该已经启动并运行了一个</font>[<font style="color:rgb(28, 30, 33);">Milvus 实例</font>](https://milvus.io/docs/install_standalone-docker.md)<font style="color:rgb(28, 30, 33);">。</font>

```python
#示例：milvus_zilliz.py
#pip install --upgrade --quiet  pymilvus
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
```

```python
vector_db = Zilliz.from_documents(  # or Milvus.from_documents
    docs,
    embeddings,
    #存储到collection_1中
    collection_name="collection_1",
    connection_args={"uri": os.getenv("MILVUS_API_URL"), "token": os.getenv("MILVUS_API_KEY")},
    #drop_old=True,  # Drop the old Milvus collection if it exists
    auto_id=True,
)
```

```python
query = "Pixar公司是做什么的?"
docs = vector_db.similarity_search(query)
```

```python
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```



## 总结
+ 下面是 Chroma、Weaviate、Qdrant 和 Milvus 四个向量数据库的功能对比

| 功能/特性 | Chroma | Weaviate | Qdrant | Milvus |
| --- | --- | --- | --- | --- |
| 数据模型 | 向量 + 元数据 | 向量 + 元数据 | 向量 + 元数据 | 向量 + 元数据 |
| 支持的索引类型 | HNSW | HNSW, IVF, Flat | HNSW | IVF, HNSW, ANNOY, Flat |
| 扩展性 | 高 | 高 | 高 | 高 |
| 实时性 | 实时更新 | 实时更新 | 实时更新 | 实时更新 |
| 多样化查询 | 向量相似性搜索 | 向量相似性搜索 + 混合查询 | 向量相似性搜索 | 向量相似性搜索 + 混合查询 |
| 分布式架构 | 是 | 是 | 是 | 是 |
| 支持的语言 | Python, JavaScript | Python, Java, Go, TypeScript | Python, Go | Python, Java, Go, Node.js |
| 社区支持 | 活跃 | 活跃 | 活跃 | 活跃 |
| 开源许可证 | <font style="color:rgb(28, 30, 33);">Apache 2.0</font> | BSD-3-Clause | Apache 2.0 | Apache 2.0 |
| 部署选项 | 本地, 云 | 本地, 云 | 本地, 云 | 本地, 云 |
| 额外功能 | 数据版本控制 | 知识图谱集成, 模型管理 | 集成向量处理工具 | 数据管理工具, 集成分析工具 |


这些数据库在功能和特性上各有优势，选择合适的数据库应根据具体的应用需求和技术栈来决定。

# FAISS, Pinecone, Lance Similarity search
## FAISS
### Faiss介绍
[<font style="color:rgb(28, 30, 33);">Facebook AI Similarity Search (Faiss /Fez/)</font>](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)<font style="color:rgb(28, 30, 33);"> 是一个用于高效相似度搜索和密集向量聚类的库。它包含了在任意大小的向量集合中进行搜索的算法，甚至可以处理可能无法完全放入内存的向量集合。它还包含用于评估和参数调整的支持代码。</font>

[<font style="color:rgb(28, 30, 33);">Faiss </font>](https://faiss.ai/)<font style="color:rgb(28, 30, 33);">官方文档：</font>[https://faiss.ai/](https://faiss.ai/)

<font style="color:rgb(28, 30, 33);">下面展示如何使用与 </font>`<font style="color:rgb(28, 30, 33);">FAISS</font>`<font style="color:rgb(28, 30, 33);"> 向量数据库相关的功能。它将展示特定于此集成的功能。在学习完这些内容后，探索</font>[<font style="color:rgb(28, 30, 33);">相关的用例页面</font>](http://www.aidoczh.com/langchain/v0.2/docs/how_to/#qa-with-rag)<font style="color:rgb(28, 30, 33);">可能会很有帮助，以了解如何将这个向量存储作为更大链条的一部分来使用。</font>

### 设置
<font style="color:rgb(28, 30, 33);">该集成位于</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">langchain-community</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">包中。我们还需要安装</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">faiss</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">包本身。我们还将使用 OpenAI 进行嵌入，因此需要安装这些要求。我们可以使用以下命令进行安装：</font>

```bash
pip install -U langchain-community faiss-cpu langchain-openai tiktoken
```

<font style="color:rgb(28, 30, 33);">请注意，如果您想使用启用了 GPU 的版本，也可以安装 </font>`<font style="color:rgb(28, 30, 33);">faiss-gpu</font>`<font style="color:rgb(28, 30, 33);">。</font>

<font style="color:rgb(28, 30, 33);">设置</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">LangSmith</font>](https://smith.langchain.com/)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">以获得最佳的可观测性也会很有帮助（但不是必需的）。</font>

```python
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""
```

### 导入
<font style="color:rgb(28, 30, 33);">在这里，我们将文档导入到向量存储中。</font>

```python
#示例：faiss_search.py
# 如果您需要使用没有 AVX2 优化的 FAISS 进行初始化，请取消下面一行的注释
# os.environ['FAISS_NO_AVX2'] = '1'
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
print(db.index.ntotal)
```

```plain
16
```

### 查询
<font style="color:rgb(28, 30, 33);">现在，我们可以查询向量存储。有几种方法可以做到这一点。最常见的方法是使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">similarity_search</font>`<font style="color:rgb(28, 30, 33);">。</font>

```python
#示例：faiss_search.py
query = "Pixar公司是做什么的?"
docs = db.similarity_search(query)
```

```python
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

### 作为检索器
<font style="color:rgb(28, 30, 33);">我们还可以将向量存储转换为</font><font style="color:rgb(28, 30, 33);"> </font>[<font style="color:rgb(28, 30, 33);">Retriever</font>](http://www.aidoczh.com/langchain/v0.2/docs/how_to/#retrievers)<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">类。这使我们能够轻松地在其他 LangChain 方法中使用它，这些方法主要用于检索器。</font>

```python
#示例：faiss_retriever.py
retriever = db.as_retriever()
docs = retriever.invoke(query)
```

```python
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

### 带分数的相似度搜索
<font style="color:rgb(28, 30, 33);">还有一些 FAISS 特定的方法。其中之一是</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">similarity_search_with_score</font>`<font style="color:rgb(28, 30, 33);">，它允许您返回文档以及查询与它们之间的距离分数。返回的距离分数是 L2 距离。因此，得分越低越好。</font>

```python
#示例：faiss_similarity.py
#返回文档以及查询与它们之间的距离分数。返回的距离分数是 L2 距离。因此，得分越低越好。
docs_and_scores = db.similarity_search_with_score(query)
print(docs_and_scores)
#还可以使用`similarity_search_by_vector`来搜索与给定嵌入向量相似的文档，该函数接受一个嵌入向量作为参数，而不是字符串。
embedding_vector = embeddings.embed_query(query)
docs_and_scores = db.similarity_search_by_vector(embedding_vector)
print(docs_and_scores)
```

```python
[(Document(metadata={'source': '../../resource/knowledge.txt'}, page_content="During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.\n在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。"), 0.3155345), (Document(metadata={'source': '../../resource/knowledge.txt'}, page_content='I was lucky – I found what I loved to do early in life. Woz and I started Apple in my parents garage when I was 20. We worked hard, and in 10 years Apple had grown from just the two of us in a garage into a billion company with over 4000 employees. We had just released our finest creation - the Macintosh - a year earlier, and I had just turned 30. And then I got fired. How can you get fired from a company you started? Well, as Apple grew we hired someone who I thought was very talented to run the company with me, and for the first year or so things went well. But then our visions of the future began to diverge and eventually we had a falling out. When we did, our Board of Directors sided with him. So at 30 I was out. And very publicly out. What had been the focus of my entire adult life was gone, and it was devastating.\n我非常幸运，因为我在很早的时候就找到了我钟爱的东西。沃兹和我在二十岁的时候就在父母的车库里面开创了苹果公司。我们工作得很努力，十年之后，这个公司从那两个车库中的穷光蛋发展到了超过四千名的雇员、价值超过二十亿的大公司。在公司成立的第九年，我们刚刚发布了最好的产品，那就是 Macintosh。我也快要到三十岁了。在那一年，我被炒了鱿鱼。你怎么可能被你自己创立的公司炒了鱿鱼呢？嗯，在苹果快速成长的时候，我们雇用了一个很有天分的家伙和我一起管理这个公司，在最初的几年，公司运转的很好。但是后来我们对未来的看法发生了分歧, 最终我们吵了起来。当争吵不可开交的时候，董事会站在了他的那一边。所以在三十岁的时候，我被炒了。在这么多人的眼皮下我被炒了。在而立之年，我生命的全部支柱离自己远去，这真是毁灭性的打击。'), 0.44481623), (Document(metadata={'source': '../../resource/knowledge.txt'}, page_content="I really didn't know what to do for a few months. I felt that I had let the previous generation of entrepreneurs down - that I had dropped the baton as it was being passed to me. I met with David Packard and Bob Noyce and tried to apologize for screwing up so badly. I was a very public failure, and I even thought about running away from the valley. But something slowly began to dawn on me – I still loved what I did. The turn of events at Apple had not changed that one bit. I had been rejected, but I was still in love. And so I decided to start over.\n在最初的几个月里，我真是不知道该做些什么。我把从前的创业激情给丢了，我觉得自己让与我一同创业的人都很沮丧。我和 David Pack 和 Bob Boyce 见面，并试图向他们道歉。我把事情弄得糟糕透顶了。但是我渐渐发现了曙光，我仍然喜爱我从事的这些东西。苹果公司发生的这些事情丝毫的没有改变这些，一点也没有。我被驱逐了，但是我仍然钟爱它。所以我决定从头再来。\n\nI didn't see it then, but it turned out that getting fired from Apple was the best thing that could have ever happened to me. The heaviness of being successful was replaced by the lightness of being a beginner again, less sure about everything. It freed me to enter one of the most creative periods of my life.\n我当时没有觉察，但是事后证明，从苹果公司被炒是我这辈子发生的最棒的事情。因为，作为一个成功者的极乐感觉被作为一个创业者的轻松感觉所重新代替：对任何事情都不那么特别看重。这让我觉得如此自由，进入了我生命中最有创造力的一个阶段。"), 0.46826816), (Document(metadata={'source': '../../resource/knowledge.txt'}, page_content="I'm pretty sure none of this would have happened if I hadn't been fired from Apple. It was awful tasting medicine, but I guess the patient needed it. Sometimes life hits you in the head with a brick. Don't lose faith. I'm convinced that the only thing that kept me going was that I loved what I did. You've got to find what you love. And that is as true for your work as it is for your lovers. Your work is going to fill a large part of your life, and the only way to be truly satisfied is to do what you believe is great work. And the only way to do great work is to love what you do. If you haven't found it yet, keep looking. Don't settle. As with all matters of the heart, you'll know when you find it. And, like any great relationship, it just gets better and better as the years roll on. So keep looking until you find it. Don't settle.\n我可以非常肯定,如果我不被苹果公司开除的话，这其中一件事情也不会发生的。这个良药的味道实在是太苦了，但是我想病人需要这个药。有些时候，生活会拿起一块砖头向你的脑袋上猛拍一下。不要失去信心，我很清楚唯一使我一直走下去的，就是我做的事情令我无比钟爱。你需要去找到你所爱的东西，对于工作是如此，对于你的爱人也是如此。你的工作将会占据生活中很大的一部分。你只有相信自己所做的是伟大的工作，你才能怡然自得。如果你现在还没有找到，那么继续找、不要停下来、全心全意的去找，当你找到的时候你就会知道的。就像任何真诚的关系，随着岁月的流逝只会越来越紧密。所以继续找，直到你找到它，不要停下来。\n\nMy third story is about death.\n我的第三个故事是关于死亡的。"), 0.4740282)]
```

### 保存和加载
<font style="color:rgb(28, 30, 33);">您还可以保存和加载 FAISS 索引。这样做很有用，因为您不必每次使用时都重新创建它。</font>

```python
#示例：faiss_save.py
#保存索引
db.save_local("faiss_index")
#读取索引
new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
docs = new_db.similarity_search(query)
```

```python
docs[0]
```

```plain
page_content='During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。' metadata={'source': '../../resource/knowledge.txt'}
```



## Pinecone
### Pinecone介绍
[<font style="color:rgb(28, 30, 33);">Pinecone</font>](https://docs.pinecone.io/docs/overview)<font style="color:rgb(28, 30, 33);"> (/ˈpaɪnˌkon/ n. 松果；松球)是一个功能广泛的向量数据库。</font>

<font style="color:rgb(28, 30, 33);">官方文档：</font>[https://docs.pinecone.io/guides/get-started/quickstart](https://docs.pinecone.io/guides/get-started/quickstart)

<font style="color:rgb(28, 30, 33);">下面展示了如何使用与 </font>`<font style="color:rgb(28, 30, 33);">Pinecone</font>`<font style="color:rgb(28, 30, 33);"> 向量数据库相关的功能。</font>

<font style="color:rgb(28, 30, 33);">设置以下环境变量以便在本文档中进行操作：</font>

+ `<font style="color:rgb(28, 30, 33);">OPENAI_API_KEY</font>`<font style="color:rgb(28, 30, 33);">：您的 OpenAI API 密钥，用于使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">OpenAIEmbeddings</font>`

```python
%pip install --upgrade --quiet  \
    langchain-pinecone \
    langchain-openai \
    langchain \
    pinecone-notebooks
```

<font style="color:rgb(28, 30, 33);">配置PINECONE_API_KEY 环境变量</font>

```powershell
setx PINECONE_API_KEY ""
```

![](https://cdn.nlark.com/yuque/0/2024/png/2424104/1724049661614-25865098-3e21-4cb5-a9c8-e2caeb852032.png)

<font style="color:rgb(28, 30, 33);">首先，让我们将我们的文本库拆分成分块的 </font>`<font style="color:rgb(28, 30, 33);">docs</font>`<font style="color:rgb(28, 30, 33);">。</font>

```python
#示例：pinecone_search.py
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
loader = TextLoader("../../resource/knowledge.txt",encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
```

<font style="color:rgb(28, 30, 33);">新创建的 API 密钥已存储在</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">PINECONE_API_KEY</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">环境变量中。我们将使用它来设置 Pinecone 客户端。</font>

```python
import os
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_api_key
import time
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=pinecone_api_key)
```

<font style="color:rgb(28, 30, 33);">接下来，让我们连接到您的 Pinecone 索引。如果名为</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">index_name</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">的索引不存在，将会被创建。</font>

```python
import time
index_name = "langchain-index"  # 如果需要，可以更改
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)
```

<font style="color:rgb(28, 30, 33);">现在我们的 Pinecone 索引已经设置好，我们可以使用</font><font style="color:rgb(28, 30, 33);"> </font>`<font style="color:rgb(28, 30, 33);">PineconeVectorStore.from_documents</font>`<font style="color:rgb(28, 30, 33);"> </font><font style="color:rgb(28, 30, 33);">将这些分块的文档作为内容进行更新。</font>

```python
from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
```

```python
query = "Pixar"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

查看index创建：[https://app.pinecone.io](https://app.pinecone.io/organizations/-O3c52_oskiguJEqQubV/projects/f087a26f-d581-4af3-a628-2a13ead9fd7c/indexes)

## Lance
### [<font style="color:rgb(28, 30, 33);">LanceDB</font>](https://lancedb.com/)介绍
[<font style="color:rgb(28, 30, 33);">Lance</font>](https://lancedb.com/)<font style="color:rgb(28, 30, 33);">（ /læns/ 长矛；执矛战士；[医]柳叶刀）是一个基于持久存储构建的用于向量搜索的开源数据库，极大地简化了嵌入式的检索、过滤和管理。完全开源。</font>

<font style="color:rgb(28, 30, 33);">官网：</font>[https://lancedb.com/](https://lancedb.com/)

官方文档：[https://lancedb.github.io/lancedb/basic/](https://lancedb.github.io/lancedb/basic/)

<font style="color:rgb(28, 30, 33);">下面展示如何使用与 </font>`<font style="color:rgb(28, 30, 33);">LanceDB</font>`<font style="color:rgb(28, 30, 33);"> 向量数据库相关的功能，基于 Lance 数据格式。</font>

```python
! pip install -U langchain-openai
```

```python
! pip install lancedb
```

```python
#示例：lance_search.py
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
documents = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
```

### 向量存储
```python
docsearch = LanceDB.from_documents(documents, embeddings)
query = "Pixar公司是做什么的?"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
```

```plain
During the next five years, I started a company named NeXT, another company named Pixar, and fell in love with an amazing woman who would become my wife. Pixar went on to create the worlds first computer animated feature film, Toy Story, and is now the most successful animation studio in the world. In a remarkable turn of events, Apple bought NeXT, I retuned to Apple, and the technology we developed at NeXT is at the heart of Apple's current renaissance. And Laurene and I have a wonderful family together.
在接下来的五年里, 我创立了一个名叫 NeXT 的公司，还有一个叫Pixar的公司，然后和一个后来成为我妻子的优雅女人相识。Pixar 制作了世界上第一个用电脑制作的动画电影——“”玩具总动员”，Pixar 现在也是世界上最成功的电脑制作工作室。在后来的一系列运转中，Apple 收购了NeXT，然后我又回到了苹果公司。我们在NeXT 发展的技术在 Apple 的复兴之中发挥了关键的作用。我还和 Laurence 一起建立了一个幸福的家庭。
```

```python
print(docs[0].metadata)
```

```python
{'source': '../../resource/knowledge.txt'}
```

---

<font style="color:rgb(28, 30, 33);">此外，要探索表格，可以将其加载到数据框中或将其保存在 csv 文件中：</font>

```python
tbl = docsearch.get_table()
print("tbl:", tbl)
pd_df = tbl.to_pandas()
pd_df.to_csv("docsearch.csv", index=False)
```

```python
tbl: LanceTable(connection=LanceDBConnection(D:\tmp\lancedb), name="vectorstore")
```



## 总结
下面是 Pinecone、FAISS 和 Lance 三个向量数据库的功能对比表格：

| 功能/特性 | Pinecone | FAISS | Lance |
| --- | --- | --- | --- |
| 数据模型 | 向量 + 元数据 | 向量 | 向量 + 元数据 |
| 支持的索引类型 | HNSW | IVF, HNSW, PQ, OPQ | HNSW |
| 扩展性 | 高 | 中 | 高 |
| 实时性 | 实时更新 | 批量处理 | 实时更新 |
| 多样化查询 | 向量相似性搜索 | 向量相似性搜索 | 向量相似性搜索 |
| 分布式架构 | 是 | 否 | 是 |
| 支持的语言 | Python, JavaScript, Go | Python, C++ | Python |
| 社区支持 | 活跃 | 活跃 | 新兴 |
| 开源许可证 | 专有 | MIT | Apache 2.0 |
| 部署选项 | 云 | 本地 | 本地, 云 |
| 额外功能 | 自动扩展, 数据管理工具 | 高度优化的搜索性能 | 数据版本控制, 集成分析工具 |


这些数据库在功能和特性上各有优势，选择合适的数据库应根据具体的应用需求和技术栈来决定。

