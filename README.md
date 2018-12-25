---
title:  "编程-LintCode"
layout: post
categories: python 编程-lintcode
tags:  python
author: 周彬
---

* content
{:toc}

# 数据结构与算法(python)


**chapter_2 二分法(Binary Search)**

第一境界（二分位置 之 OOXX）

 - [二分查找](#二分查找)
 - [第一个错误的代码版本](#第一个错误的代码版本)
 - [在大数组中查找](#在大数组中查找)
 - [寻找旋转排序数组中的最小值](#寻找旋转排序数组中的最小值)
 - [搜索二维矩阵](#搜索二维矩阵)
 - [搜索二维矩阵 II](#搜索二维矩阵2)
 - [搜索区间](#搜索区间)
 - [目标出现总和](#目标出现总和)
 - [山脉序列中的最大值](#山脉序列中的最大值)

第二境界（二分位置 之 Half half）

 - [寻找峰值](#寻找峰值)
 - [搜索旋转排序数组star](#搜索旋转排序数组)
 
第三境界（二分答案）

 - [x的平方根](#x的平方根)
 - [木材加工](#木材加工)
 - [书籍复印](#书籍复印)
 - [两个整数相除](#两个整数相除)

**chapter_3 二叉树与分治法(Binary Tree & Divide Conquer)**

 - [二叉树的前序遍历](#二叉树的前序遍历)
 - [二叉树的中序遍历](#二叉树的中序遍历)
 - [二叉树的后序遍历](#二叉树的后序遍历)

破枪式（碰到二叉树的问题，就想想整棵树在该问题上的结果
和左右儿子在该问题上的结果之间的联系是什么）

 - [二叉树的最大深度](#二叉树的最大深度)
 - [二叉树的所有路径](#二叉树的所有路径)
 - [最小子树](#最小子树)
 - [平衡二叉树](#平衡二叉树)
 - [具有最大平均数的子树](#具有最大平均数的子树)
 - [将二叉树拆成链表](#将二叉树拆成链表)
 - [Lowest Common Ancestor of a Binary Tree](#lowest_common_ancestor_of_a_binary_tree)
 - [二叉树最长连续序列](#二叉树最长连续序列)
 - [二叉树的路径和](#二叉树的路径和)
 - [二叉树的路径和 II](#二叉树的路径和2)
 - [验证二叉查找树](#验证二叉查找树)
 - [二叉查找树迭代器](#二叉查找树迭代器)
 - [前序遍历和中序遍历树构造二叉树](#前序遍历和中序遍历树构造二叉树)
 - [二叉树的最小深度](#二叉树的最小深度)
 - [Same Tree](#same_tree)
 - [翻转二叉树](#翻转二叉树)
 
**chapter_4 宽度优先搜索(Breadth First Search)**

什么时候应该用BFS？<br>

图的遍历 Traversal in Graph
（1. 层级遍历 Level Order Traversal； 2. 由点及面 Connected Component；3. 拓扑排序 Topological Sorting）

最短路径 Shortest Path in Simple Graph<br>
（仅限简单图求最短路径，即图中每条边长度都是1，且没有方向）

二叉树上的宽度优先搜索(BFS in Binary Tree)

 - [二叉树的层次遍历](#二叉树的层次遍历)
 - [二叉树的序列化和反序列化](#二叉树的序列化和反序列化)
 - [将二叉树按照层级转化为链表](#将二叉树按照层级转化为链表)

图上的宽度优先搜索(BFS in Graph)

 - [克隆图](#克隆图)
 - [搜索图中节点](#搜索图中节点)
 - [拓扑排序](#拓扑排序)
 - [课程表](#课程表)
 - [找无向图的连通块](#找无向图的连通块)
 - [单词接龙](#单词接龙)

矩阵中的宽度优先搜索(BFS in Matrix)

 - [岛屿的个数](#岛屿的个数)
 - [僵尸矩阵](#僵尸矩阵)
 - [骑士的最短路线](#骑士的最短路线)
 
**chapter_5 深度优先搜索(Depth First Search)**

碰到让你找所有方案的题，一定是DFS<br>
90%DFS的题，要么是排列，要么是组合<br>

 - [子集](#子集)
 - [数字组合](#数字组合)
 - [数字组合 II](#数字组合2)
 - [分割回文串](#分割回文串)
 - [全排列](#全排列)
 - [全排列 II(带重复元素的排列)](全排列2)
 - [N皇后问题](#n皇后问题)
 - [字符串解码](#字符串解码)
 - [摊平嵌套的列表](#摊平嵌套的列表)

栈相关的问题<br>

 - [用栈实现队列](#用栈实现队列)
 - [直方图最大矩形覆盖](#直方图最大矩形覆盖)
 - [带最小值操作的栈 min-stack](#带最小值操作的栈)

总结 Conclusion<br>
什么时候用 DFS？ 求所有方案时<br>
怎么解决DFS？ 不是排列就是组合<br>
复杂度怎么算？ O(答案个数 * 构造每个答案的时间复杂度)<br>
非递归怎么办？ 必“背”程序<br>

**chapter_6 链表与数组(Linked List & Array)**

Linked List

 - [翻转链表](#翻转链表)
 - [翻转链表 II](#翻转链表2)
 - [K组翻转链表](#K组翻转链表)
 - [链表划分](#链表划分)
 - [合并两个排序链表](#合并两个排序链表)
 - [交换链表当中两个节点](#交换链表当中两个节点)
 - [重排链表](#重排链表)
 - [旋转链表](#旋转链表)
 - [带环链表](#带环链表)
 - [带环链表 II](#带环链表2)
 
未做（高频题）：<br>
 - [复制带随机指针的链表](#复制带随机指针的链表)
 - [链表排序](#链表排序)

Array<br>
技巧一：用prefix_sum<br>
技巧二：

 - [最大子数组](#最大子数组)
 - [子数组之和](#子数组之和)
 - [最接近零的子数组和](#最接近零的子数组和)
 - [整数排序 II(quick sort&merge sort)](#整数排序2)
 - [合并排序数组](#合并排序数组)
 - [合并排序数组 II](#合并排序数组2)
 - [两个排序数组的中位数](#两个排序数组的中位数)
 - [买卖股票的最佳时机](#买卖股票的最佳时机)
 - [买卖股票的最佳时机 II](#买卖股票的最佳时机2)
 - [最长连续序列](#最长连续序列)
 
 
**chapter_7 两根指针(Two Pointers)**

 - [移动零](#移动零)
 - [去除重复元素](#去除重复元素)
 - [有效回文串](#有效回文串)
 - [数组划分](#数组划分)
 - [无序数组K小元素](#无序数组K小元素)
 - [交错正负数](#交错正负数)
 - [字符大小写排序](#字符大小写排序)
 - [两数之和](#两数之和)
 - [两数之和 - 不同组成](#两数之和_不同组成)
 - [三数之和](#三数之和)
 - [两数和-小于或等于目标值](#两数和_小于或等于目标值)
 - [最接近的三数之和](#最接近的三数之和)
 - [四数之和](#四数之和)
 - [两数和 - 差等于目标值](#两数和_差等于目标值)

**chapter_8 数据结构(Data Structure)**

独孤九剑 —— 破箭式<br>
BFS的主要数据结构是Queue<br>
DFS的主要数据结构是Stack<br>
千万不要搞反了！很体现基础知识的扎实度！<br>

 - [字符串解码](#字符串解码)
 - [用栈实现队列](#用栈实现队列)
 - [双队列实现栈](#双队列实现栈)
 - [重哈希](#重哈希)
 - [LRU缓存策略](#LRU缓存策略)
 - [乱序字符串](#乱序字符串)
 - [堆化 Heapify](#堆化)
 - [丑数 II](#丑数2)
 - [最高频的K个单词](#最高频的k个单词)

**chapter_9 动态规划(Dynamic Programming)**

通过一道经典题理解动态规划<br>
1.递归与动规的联系与区别<br>
2.记忆化搜索
 - [数字三角形](#数字三角形)

什么时候使用动态规划<br>

---

### 二分查找

[二分查找问题描述](http://www.lintcode.com/problem/first-position-of-target/)<br>
```python
class Solution:
    """
    @param nums: The integer array.
    @param target: Target to find.
    @return: The first position of target. Position starts from 0.
    """
    def binarySearch(self, nums, target):
        # write your code here
        if not nums or len(nums) == 0:
            return -1
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            # 细节，不写成(start + end) / 2：为了防止溢出2^32
            mid = int((end - start) / 2 + start)
            # mid = ((end - start) >> 1) + start #注意python运算符和优先级
            if target == nums[mid]:
                end = mid
            elif target < nums[mid]:
                end = mid
            else:
                start = mid
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1
```

### 第一个错误的代码版本

[第一个错误的代码版本](https://www.lintcode.com/problem/first-bad-version/description)<br>
```python
#class SVNRepo:
#    @classmethod
#    def isBadVersion(cls, id)
#        # Run unit tests to check whether verison `id` is a bad version
#        # return true if unit tests passed else false.
# You can use SVNRepo.isBadVersion(10) to check whether version 10 is a
# bad version.
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        # write your code here
        if n < 1:
            return -1
        if n == 1 and SVNRepo.isBadVersion(1):
            return 1
        start = 1
        end = n
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid
        if SVNRepo.isBadVersion(start):
            return start
        if SVNRepo.isBadVersion(end):
            return end
        return -1
```

### 在大数组中查找

[在大数组中查找](https://www.lintcode.com/problem/search-in-a-big-sorted-array/description)<br>
```python
class Solution:
    """
    @param: reader: An instance of ArrayReader.
    @param: target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        # write your code here
        if not reader:
            return -1
        start = 0
        end = 1
        while reader.get(end) < target:
            end *= 2
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if reader.get(mid) == target:
                end = mid
            elif reader.get(mid) > target:
                end = mid
            else:
                start = mid
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1
```

### 寻找旋转排序数组中的最小值

[寻找旋转排序数组中的最小值](https://www.lintcode.com/problem/find-minimum-in-rotated-sorted-array/description)<br>
```python
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return -9999
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if nums[mid] > nums[end]:
                start = mid
            else:
                end = mid
        if nums[start] < nums[end]:
            return nums[start]
        return nums[end]
```

### 搜索二维矩阵

[搜索二维矩阵](https://www.lintcode.com/problem/search-a-2d-matrix/description)<br>
```python
class Solution:
    """
    @param matrix: matrix, a list of lists of integers
    @param target: An integer
    @return: a boolean, indicate whether matrix contains target
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix or len(matrix) == 0:
            return False
        col = 0
        row = len(matrix) - 1
        while col < len(matrix[0]) and row > -1:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] < target:
                col += 1
            else:
                row -= 1
        return False
```

### 搜索二维矩阵2

[搜索二维矩阵 II](https://www.lintcode.com/problem/search-a-2d-matrix-ii/description)<br>
```python
class Solution:
    """
    @param matrix: A list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicate the total occurrence of target in the given matrix
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix or len(matrix) == 0:
            return 0
        col = 0
        row = len(matrix) - 1
        res = 0
        while col < len(matrix[0]) and row > -1:
            if matrix[row][col] == target:
                res += 1
                row -= 1
            elif matrix[row][col] < target:
                col += 1
            else:
                row -= 1
        return res
```

### 搜索区间

[搜索区间](https://www.lintcode.com/problem/search-for-a-range/description)
思路：先搜索target最左边的值，然后再把这个值赋给start重新搜索最右边的值<br>
```python
class Solution:
    """
    @param A : a list of integers
    @param target : an integer to be searched
    @return : a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        if not A or len(A) == 0:
            return [-1, -1]
        
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if A[mid] < target:
                start = mid
            else:
                end = mid
        
        if A[start] == target:
            leftBound = start
        elif A[end] == target:
            leftBound = end
        else:
            return [-1, -1]
        
        start, end = leftBound, len(A) - 1
        while start + 1 < end:
            mid = int((end - start) / 2 + start)
            if A[mid] <= target:
                start = mid
            else:
                end = mid
        if A[end] == target:
            rightBound = end
        else:
            rightBound = start
        return [leftBound, rightBound]
```

### 山脉序列中的最大值

[山脉序列中的最大值](https://www.lintcode.com/problem/maximum-number-in-mountain-sequence/description)<br>
时间复杂度O(n)<br>
```python
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return -9999
        if len(nums) == 1:
            return nums[0]
        i = 0
        j = 1
        while j < len(nums):
            if nums[i] > nums[j]:
                return nums[i]
            i += 1
            j += 1
        return -9999
```

时间复杂度O(logn)<br>
```python
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return -9999
        if len(nums) == 1:
            return nums[0]
        start = 0
        end = len(nums) - 1
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if nums[mid] < nums[mid + 1]:
                start = mid
            else:
                end = mid
        if nums[start] > nums[end]:
            return nums[start]
        else:
            return nums[end]
```

### 寻找峰值

[寻找峰值](https://www.lintcode.com/problem/find-peak-element/description)<br>
```python
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        if not A or len(A) < 3:
            return -1
        start = 0
        end = len(A) - 1
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if A[mid] > A[mid - 1] and A[mid] > A[mid + 1]:
                return mid
            elif A[mid] > A[mid - 1] and A[mid] < A[mid + 1]:
                start = mid
            else:
                end = mid
        return -1
```

### 搜索旋转排序数组

[搜索旋转排序数组](https://www.lintcode.com/problem/search-in-rotated-sorted-array/description)<br>
```python
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        if not A or len(A) == 0:
            return -1
        start = 0
        end = len(A) - 1
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if A[mid] == target:
                return mid
            # 画图 看纸质ppt
            if A[start] < A[mid]:
                if A[start] <= target and target <= A[mid]:
                    end = mid
                else:
                    start = mid
            else:
                if A[mid] <= target and target <= A[end]:
                    start = mid
                else:
                    end = mid
        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1
```

### x的平方根

[x的平方根](https://www.lintcode.com/problem/sqrtx/)<br>
```python
class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """
    def sqrt(self, x):
        # write your code here
        if x < 1:
            return 0
        if x == 1:
            return 1
        start = 1
        end = x
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if mid * mid <= x and (mid+1)*(mid+1) >= x:
                return mid
            elif mid * mid > x:
                end = mid
            else:
                start = mid
        return 0
```

### 木材加工

[木材加工](https://www.lintcode.com/problem/wood-cut/description)<br>
```python
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        # write your code here
        if not L or len(L) == 0 or k < 1:
            return 0
        start = 1
        end = max(L)
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            r = sum([x//mid for x in L])
            if r == k:
                start = mid
            elif r > k:
                start = mid
            else:
                end = mid
        print(start)
        print(end)
        if sum([x//end for x in L]) >= k:
            return end
        if sum([x//start for x in L]) >= k:
            return start
        return 0
```

### 书籍复印

[书籍复印](https://www.lintcode.com/problem/copy-books/description)<br>
一句话描述题意：将数组切分为k个子数组，让数组和最大的最小<br>
```
令狐冲解法：
使用九章算法强化班中讲过的基于答案值域的二分法。
答案的范围在 max(pages)~sum(pages) 之间，每次二分到一个时间 time_limit 的时候，用贪心法从左到右扫描一下 pages，看看需要多少个人来完成抄袭。
如果这个值 <= k，那么意味着大家花的时间可能可以再少一些，如果 > k 则意味着人数不够，需要降低工作量。

时间复杂度 O(nlog(sum))O(nlog(sum)) 是该问题时间复杂度上的最优解法
```
```python
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        # write your code here
        if not pages or len(pages) == 0 or k < 1:
            return 0
        start = max(pages)
        end = sum(pages)
        if k == 1:
            return end
        while start + 1 < end:
            mid = ((end - start) >> 1) + start
            if self.check(pages, mid) <= k:
                end = mid
            else:
                start = mid
        if self.check(pages, start) <= k:
            return start
        return end
        
    def check(self, pages, time_limit):
        count = 1
        cul_pages = 0
        for p in pages:
            if cul_pages + p > time_limit:
                count += 1
                cul_pages = 0
            cul_pages += p
        return count
```

### 两个整数相除

[两个整数相除](https://www.lintcode.com/problem/divide-two-integers/description)
```python
class Solution:
    """
    @param dividend: the dividend
    @param divisor: the divisor
    @return: the result
    """
    def divide(self, dividend, divisor):
        # write your code here
        INT_MAX = 2147483647
        if divisor == 0:
            return INT_MAX
        if dividend == 0:
            return 0
        neg = (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0)
        dividend, divisor = abs(dividend), abs(divisor)
        res, shift = 0, 31
        while shift >= 0:
            # 100 >= (9<<3)才会进到if里面
            if dividend >= (divisor << shift):
                dividend -= (divisor << shift)
                res += (1 << shift)
            shift -= 1
        if neg:
            res = -res
        if res > INT_MAX:
            return INT_MAX
        return res
```

--- 

### 二叉树的前序遍历

[二叉树的前序遍历](https://www.lintcode.com/problem/binary-tree-preorder-traversal/description)<br>
```python
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
# version1 Recursion(Divide & Conquer)
class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        # 根左右
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

# version2 Recursion(Traverse)
class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        res = []
        if root is None:
            return res
        self.helper(root, res)
        
        return res
    
    def helper(self, root, res):
        if root is None:
            return 
        res.append(root.val)
        self.helper(root.left, res)
        self.helper(root.right, res)

# version3 Non-Recursion
class Solution:
    """
    @param root: The root of binary tree.
    @return: Preorder in list which contains node values.
    """
    def preorderTraversal(self, root):
        if root is None:
            return []
        stack = [root]
        preorder = []
        while stack:
            node = stack.pop()
            preorder.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return preorder
```
### 二叉树的中序遍历

[二叉树的中序遍历](https://www.lintcode.com/problem/binary-tree-inorder-traversal/description)<br>
```python
# Divide & Conquer
class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        # 左根右
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

### 二叉树的后序遍历

[二叉树的后序遍历](https://www.lintcode.com/problem/binary-tree-postorder-traversal/description)<br>
```python
# version1 Divide & Conquer
class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here
        if root is None:
            return []
        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

# version2 Traverse
class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here
        res = []
        if root is None:
            return res
        self.helper(root, res)
        return res
    def helper(self, root, res):
        if root is None:
            return
        self.helper(root.left, res)
        self.helper(root.right, res)
        res.append(root.val)
```

### 二叉树的最大深度

[二叉树的最大深度](https://www.lintcode.com/problem/maximum-depth-of-binary-tree/description)<br>
```python
# Divide & Conquer
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxDepth(self, root):
        # write your code here
        if root is None:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1
```

### 二叉树的所有路径

[二叉树的所有路径](https://www.lintcode.com/problem/binary-tree-paths/description)<br>
```python
class Solution:
    """
    @param root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        # write your code here
        if root is None:
            return []
        if root.left is None and root.right is None:
            return [str(root.val)]
        left = self.binaryTreePaths(root.left)
        right = self.binaryTreePaths(root.right)
        
        paths = []
        for p in left:
            paths.append(str(root.val) + '->' + p)
        for p in right:
            paths.append(str(root.val) + '->' + p)
        return paths
```

### 最小子树

[最小子树](https://www.lintcode.com/problem/minimum-subtree/description)<br>
```python
# Traverse + Divide Conquer
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        if root is None:
            return None
        self.min_weight = sys.maxsize
        self.res = None
        self.helper(root)
        return self.res
    def helper(self, root):
        if root is None:
            return 0
        left_weight = self.helper(root.left)
        right_weight = self.helper(root.right)
        
        tmp_sum = left_weight + right_weight + root.val
        if tmp_sum < self.min_weight:
            self.min_weight = tmp_sum
            self.res = root
        return tmp_sum
```

### 平衡二叉树

[平衡二叉树](https://www.lintcode.com/problem/balanced-binary-tree/description)<br>
```python
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        if root is None:
            return True
        isbal, _ = self.helper(root)
        return isbal
        
    def helper(self, root):
        if root is None:
            return True, 0
        isbal_left, height_left = self.helper(root.left)
        if not isbal_left:
            return False, 0
        isbal_right, height_right = self.helper(root.right)
        if not isbal_right:
            return False, 0
        return abs(height_left - height_right) <= 1, max(height_left, height_right) + 1
```

### 具有最大平均数的子树

[具有最大平均数的子树](https://www.lintcode.com/problem/subtree-with-maximum-average/description)<br>
```python
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """
    def findSubtree2(self, root):
        # write your code here
        if root is None:
            return None
        self.max_avg = -sys.maxsize
        self.res = None
        self.helper(root)
        return self.res
        
    def helper(self, root):
        if root is None:
            return 0, 0
        left_sum, left_count = self.helper(root.left)
        right_sum, right_count = self.helper(root.right)
        tmp_avg = (left_sum + right_sum + root.val) / (left_count + right_count + 1)
        if tmp_avg > self.max_avg:
            self.max_avg = tmp_avg
            self.res = root
        return left_sum + right_sum + root.val, left_count + right_count + 1
```

### 将二叉树拆成链表

[将二叉树拆成链表](https://www.lintcode.com/problem/flatten-binary-tree-to-linked-list/description)<br>
```python
# 我的思路：用前序遍历一遍node存在list中，然后依次连接left=None,right=next
# 需要使用额外的空间耗费(挑战)
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    res = []
    def flatten(self, root):
        # write your code here
        if root is None:
            return
        self.helper(root)
        for i in range(len(self.res)-1):
            self.res[i].left = None
            self.res[i].right = self.res[i+1]
        root = self.res[0]
        
    def helper(self, root):
        if root is None:
            return
        self.res.append(root)
        self.helper(root.left)
        self.helper(root.right)

# 令狐冲老师代码（不使用额外的空间耗费）
# Divide & Conquer
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        # write your code here
        if root is None:
            return
        self.helper(root)
    def helper(self, root):
        if root is None:
            return None
        left = self.helper(root.left)
        right = self.helper(root.right)
        if left is not None:
            left.right = root.right
            root.right = root.left
            root.left = None
        if right is not None:
            return right
        if left is not None:
            return left
        return root
```

### Lowest_Common_Ancestor_of_a_Binary_Tree

[Lowest Common Ancestor of a Binary Tree](https://www.lintcode.com/problem/lowest-common-ancestor-of-a-binary-tree/description)<br>
```python
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        if root is None:
            return None
        if root == A or root == B:
            return root
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        if left and right:
            return root
        if left:
            return left
        if right:
            return right
        return None
```

### 二叉树最长连续序列

[二叉树最长连续序列](https://www.lintcode.com/problem/binary-tree-longest-consecutive-sequence/description)<br>
```python
class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive(self, root):
        # write your code here
        return self.helper(root, None, 0)
    def helper(self, root, parent, lengthwithoutroot):
        if root is None:
            return 0
        # 加上root以后的看能形成的长度
        length = lengthwithoutroot + 1 if (parent is not None and parent.val + 1 == root.val) else 1
        left = self.helper(root.left, root, length)
        right = self.helper(root.right, root, length)
        return max(length, max(left, right))
```

### 二叉树的路径和

[二叉树的路径和](https://www.lintcode.com/problem/binary-tree-path-sum/description)<br>
```python
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    res = []
    def binaryTreePathSum(self, root, target):
        # write your code here
        if root is None:
            return []
        path = []
        self.helper(root, path, target)
        return self.res
    def helper(self, root, path, target):
        if root is None:
            return []
        path.append(root.val)
        if root.left is None and root.right is None:
            if sum(path) == target:
                self.res.append(path.copy())
        self.helper(root.left, path.copy(), target)
        self.helper(root.right, path.copy(), target)
```

### 二叉树的路径和2

[二叉树的路径和2](https://www.lintcode.com/problem/binary-tree-path-sum-ii/description)<br>
```python
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    res = []
    def binaryTreePathSum2(self, root, target):
        # write your code here
        if root is None:
            return []
        self.helper(root, target, [], 0)
        return self.res
    def helper(self, root, target, path, l):
        if root is None:
            return
        path.append(root.val)
        tmp = target
        for i in range(l, -1, -1):
            tmp -= path[i]
            if tmp == 0:
                self.res.append(path[i:].copy())
        self.helper(root.left, target, path, l+1)
        self.helper(root.right, target, path, l+1)
        path.pop()
```

### 验证二叉查找树

[验证二叉查找树](https://www.lintcode.com/problem/validate-binary-search-tree/description)<br>
```python
class resulttype:
    def __init__(self, isbst, max_val, min_val):
        self.isbst = isbst
        self.max_val = max_val
        self.min_val = min_val

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        res = self.helper(root)
        return res.isbst
    def helper(self, root):
        if root is None:
            return resulttype(True, -sys.maxsize, sys.maxsize)
        left = self.helper(root.left)
        right = self.helper(root.right)
        if (not left.isbst or not right.isbst):
            return resulttype(False, 0, 0)
        if (left is not None and left.max_val >= root.val) or (right is not None and right.min_val <= root.val):
            return resulttype(False, 0, 0)
        return resulttype(True, max(root.val, right.max_val), min(root.val, left.min_val))
```

### 二叉查找树迭代器

[二叉查找树迭代器](https://www.lintcode.com/problem/binary-search-tree-iterator/description)<br>
```python
class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    """
    @return: True if there has next node, or false
    """
    def hasNext(self):
        return len(self.stack) > 0

    """
    @return: return next node
    """
    def next(self):
        node = self.stack[-1]
        if node.right:
            n = node.right
            while n:
                self.stack.append(n)
                n = n.left
        else:
            n = self.stack.pop()
            while self.stack and self.stack[-1].right == n:
                n = self.stack.pop()
        return node
```

### 前序遍历和中序遍历树构造二叉树

[前序遍历和中序遍历树构造二叉树]()<br>
```python
class Solution:
    """
    @param inorder: A list of integers that inorder traversal of a tree
    @param postorder: A list of integers that postorder traversal of a tree
    @return: Root of a tree
    """
    def buildTree(self, preorder, inorder):
        # write your code here
        if len(preorder) != len(inorder):
            return None
        return self.helper(inorder, 0, len(inorder)-1, preorder, 0, len(preorder)-1)
    def helper(self, inorder, instart, inend, preorder, prestart, preend):
        if instart > inend:
            return None
        root = TreeNode(preorder[prestart])
        position = self.findPosition(inorder, instart, inend, preorder[prestart])
        
        root.left = self.helper(inorder, instart, position-1, preorder, prestart+1, prestart+position-instart)
        root.right = self.helper(inorder, position+1, inend, preorder, preend+position-inend+1, preend)
        return root
    def findPosition(self, inorder, start, end, key):
        i = start
        while i <= end:
            if inorder[i] == key:
                return i
            i += 1
        return -1
```

### 二叉树的最小深度

[二叉树的最小深度](https://www.lintcode.com/problem/minimum-depth-of-binary-tree/description)<br>
```python
class Solution:
    """
    @param root: The root of binary tree
    @return: An integer
    """
    min_depth = sys.maxsize
    def minDepth(self, root):
        # write your code here
        if not root:
            return 0
        self.helper(root, 0)
        return self.min_depth + 1
    def helper(self, root, depth):
        if not root:
            return 0
        if not root.left and not root.right:
            if self.min_depth > depth:
                self.min_depth = depth
        self.helper(root.left, depth + 1)
        self.helper(root.right, depth + 1)
```

### Same_Tree

[Same Tree](https://www.lintcode.com/problem/same-tree/description)<br>
```python
class Solution:
    """
    @param a: the root of binary tree a.
    @param b: the root of binary tree b.
    @return: true if they are identical, or false.
    """
    def isIdentical(self, a, b):
        # write your code here
        if not a and not b:
            return True
        if not a or not b:
            return False
        if a.val != b.val:
            return False
        left = self.isIdentical(a.left, b.left)
        right = self.isIdentical(a.right, b.right)
        return left and right
```

### 翻转二叉树

[翻转二叉树](https://www.lintcode.com/problem/invert-binary-tree/description)<br>
```python
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        # write your code here
        if not root:
            return None
        self.helper(root)
    def helper(self, root):
        if not root:
            return
        left = root.left
        right = root.right
        root.left = right
        root.right = left
        self.helper(root.left)
        self.helper(root.right)
```

---

### 二叉树的层次遍历

[二叉树的层次遍历](https://www.lintcode.com/problem/binary-tree-level-order-traversal/description)<br>
```python
from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        # write your code here
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        ## 二叉树的层次遍历 II
        # return res[::-1]
        return res
```

### 二叉树的序列化和反序列化

[二叉树的序列化和反序列化](https://www.lintcode.com/problem/serialize-and-deserialize-binary-tree/description)<br>
```python
from collections import deque

class Solution:
    """
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm 
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    """
    def serialize(self, root):
        # write your code here
        if not root:
            return '{}'
        queue = deque([root])
        res = []
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if not node:
                    res.append('#')
                    continue
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
        return '{%s}' % ','.join(res)

    """
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    """
    def deserialize(self, data):
        # write your code here
        data = data.strip()
        if data == '{}':
            return None
        vals = data[1:-1].split(',')
        root = TreeNode(int(vals[0]))
        queue = [root]
        isLeftchild = True
        index = 0
        for val in vals[1:]:
            if val != '#':
                node = TreeNode(int(val))
                if isLeftchild:
                    queue[index].left = node
                else:
                    queue[index].right = node
                queue.append(node)
            if not isLeftchild:
                index += 1
            isLeftchild = not isLeftchild
        return root
```

### 将二叉树按照层级转化为链表

[将二叉树按照层级转化为链表](https://www.lintcode.com/problem/convert-binary-tree-to-linked-lists-by-depth/description)<br>
```python
from collections import deque

class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
    def binaryTreeToLists(self, root):
        # Write your code here
        if not root:
            return []
        queue = deque([root])
        res = []
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(ListNode(node.val))
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(level)
        # 把列表中的listnode连接起来
        for sub_arr in res:
            if len(sub_arr) == 1:
                continue
            for index in range(len(sub_arr)-1):
                sub_arr[index].next = sub_arr[index+1]
        # return每个链表的头
        return [x[0] for x in res]
```

### 克隆图

[克隆图](https://www.lintcode.com/problem/clone-graph/description)<br>
```python
from collections import deque

class Solution:
    """
    @param: node: A undirected graph node
    @return: A undirected graph node
    """
    def cloneGraph(self, node):
        # write your code here
        if not node:
            return None
        root = node
        
        nodes = self.getNodes(node)
        
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        return mapping[root]
    def getNodes(self, node):
        queue = deque([node])
        res = set([node])
        while queue:
            node = queue.popleft()
            for neighbor in node.neighbors:
                if neighbor not in res:
                    res.add(neighbor)
                    queue.append(neighbor)
        return res
```

### 搜索图中节点

[搜索图中节点](https://www.lintcode.com/problem/search-graph-nodes/description)<br>
```python
from collections import deque

class Solution:
    """
    @param: graph: a list of Undirected graph node
    @param: values: a hash mapping, <UndirectedGraphNode, (int)value>
    @param: node: an Undirected graph node
    @param: target: An integer
    @return: a node
    """
    def searchNode(self, graph, values, node, target):
        # write your code here
        if not node:
            return None
        queue = deque([node])
        while queue:
            node = queue.popleft()
            if values[node] == target:
                return node
            for neighbor in node.neighbors:
                queue.append(neighbor)
        return None
```

### 拓扑排序

[拓扑排序](https://www.lintcode.com/problem/topological-sorting/description)<br>
```python
from collections import deque

class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        if not graph:
            return []
        node_map = self.get_indegree(graph)
        
        res = []
        start_nodes = [x for x in graph if node_map[x] == 0]
        queue = deque(start_nodes)
        while queue:
            node = queue.popleft()
            res.append(node)
            for neighbor in node.neighbors:
                node_map[neighbor] -= 1
                if node_map[neighbor] == 0:
                    queue.append(neighbor)
        return res
        
    def get_indegree(self, graph):
        node_map = {x:0 for x in graph}
        
        for node in graph:
            for neighbor in node.neighbors:
                node_map[neighbor] += 1
        return node_map
```

### 课程表

[课程表](https://www.lintcode.com/problem/course-schedule/description)<br>
```python
class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        # write your code here
        pre = {i : [] for i in range(numCourses)}
        degrees = [0] * numCourses
        for i, j in prerequisites:
            pre[j].append(i)
            degrees[i] += 1
        
        q = [i for i in range(numCourses) if degrees[i] == 0]
        res = []
        while q:
            node = q.pop()
            res.append(node)
            for i in pre[node]:
                degrees[i] -= 1
                if degrees[i] == 0:
                    q.append(i)
        return len(res) == numCourses
```

### 岛屿的个数

[岛屿的个数](https://www.lintcode.com/problem/number-of-islands/description)<br>
```python
from collections import deque

class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 0
        islands = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    self.bfs(grid, i, j)
                    islands += 1
        return islands
    def bfs(self, grid, x, y):
        queue = deque([(x, y)])
        grid[x][y] = False
        while queue:
            x, y = queue.popleft()
            for delta_x, delta_y in [(1,0),(0,1),(-1,0),(0,-1)]:
                next_x = x + delta_x
                next_y = y + delta_y
                if not self.is_val(grid, next_x, next_y):
                    continue
                queue.append((next_x, next_y))
                grid[next_x][next_y] = False
    def is_val(self, grid, x, y):
        n, m = len(grid), len(grid[0])
        return (0 <= x < n) and (0 <= y < m) and grid[x][y]
```

### 僵尸矩阵

[僵尸矩阵](https://www.lintcode.com/problem/zombie-in-matrix/description)<br>
```python
class Solution:
    """
    @param grid: a 2D integer grid
    @return: an integer
    """
    def zombie(self, grid):
        # write your code here
        n, m = len(grid), len(grid[0])
        if n == 0 or m == 0:
            return 0
        q = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    q.append((i, j))
        d = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        days = 0
        while q:
            days += 1
            new_q = []
            for node in q:
                for k in range(4):
                    x = node[0] + d[k][0]
                    y = node[1] + d[k][1]
                    if 0 <= x < n and 0 <= y < m and grid[x][y] == 0:
                        grid[x][y] = 1
                        new_q.append((x, y))
            q = new_q
            
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    return -1
        return days - 1
```

### 骑士的最短路线

[骑士的最短路线](https://www.lintcode.com/problem/knight-shortest-path/description)<br>
```python
from collections import deque
class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    def shortestPath(self, grid, source, destination):
        # write your code here
        n = len(grid)
        m = len(grid[0])
        q = deque()
        q.append([source.x, source.y])
        directions = [[1,2],[1,-2],[-1,2],[-1,-2],[2,1],[2,-1],[-2,1],[-2,-1]]
        d = 0
        while q:
            for i in range(len(q)):
                x, y = q.popleft()
                if x == destination.x and y == destination.y:
                    return d
                for i, j in directions:
                    nx, ny = x + i, y + j
                    if nx < 0 or nx >= n:
                        continue
                    if ny < 0 or ny >= m:
                        continue
                    if grid[nx][ny] == 1:
                        continue
                    q.append([nx, ny])
                    grid[nx][ny] = 1
            d += 1
        return -1
```

### 找无向图的连通块

[找无向图的连通块](https://www.lintcode.com/problem/connected-component-in-undirected-graph/description)<br>
```python
from collections import deque
class Solution:
    """
    @param: nodes: a array of Undirected graph node
    @return: a connected set of a Undirected graph
    """
    
    visited = set()
    
    def connectedSet(self, nodes):
        # write your code here
        if not nodes:
            return []
        res = []
        for node in nodes:
            if node not in self.visited:
                res.append(self.bfs(node))
        return res
    def bfs(self, node):
        sub_res = []
        q = deque([node])
        self.visited.add(node)
        while q:
            node = q.popleft()
            sub_res.append(node.label)
            for neighbor in node.neighbors:
                if neighbor in self.visited:
                    continue
                q.append(neighbor)
                self.visited.add(neighbor)
        return sorted(sub_res)
```

### 单词接龙

[单词接龙](https://www.lintcode.com/problem/word-ladder/description)<br>
```python
from collections import deque
class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: An integer
    """
    def ladderLength(self, start, end, dict):
        # write your code here
        dict.add(end)
        queue = deque([start])
        visited = set([start])
        distance = 0
        
        while queue:
            distance += 1
            for i in range(len(queue)):
                word = queue.popleft()
                if word == end:
                    return distance
                for next_word in self.get_next_word(word):
                    if next_word not in dict or next_word in visited:
                        continue
                    queue.append(next_word)
                    visited.add(next_word)
        return 0
    def get_next_word(self, word):
        next_words_list = []
        for i in range(len(word)):
            left, right = word[:i], word[i+1:]
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char == word[i]:
                    continue
                next_words_list.append(left+char+right)
        return next_words_list
```

---

### 子集

[子集](https://www.lintcode.com/problem/subsets/description)<br>
```python
class Solution:
    
    def search(self, nums, S, index):
        if index == len(nums):
            self.results.append(list(S))
            return
        
        S.append(nums[index])
        self.search(nums, S, index + 1)
        S.pop()
        self.search(nums, S, index + 1)
        
    def subsets(self, nums):
        self.results = []
        self.search(sorted(nums), [], 0)
        return self.results
```

### 数字组合

[数字组合](https://www.lintcode.com/problem/combination-sum/description)<br>
```python
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, candidates, target):
        # write your code here
        if not candidates:
            return [[]]
        candidates = sorted(list(set(candidates)))
        self.res = []
        self.dfs(candidates, target, 0, [])
        return self.res
    def dfs(self, candidates, target, start, combination):
        if target == 0:
            self.res.append(list(combination))
        for i in range(start, len(candidates)):
            if target < candidates[i]:
                return
            combination.append(candidates[i])
            self.dfs(candidates, target - candidates[i], i, combination)
            combination.pop()
```

### 数字组合2

[数字组合 II](https://www.lintcode.com/problem/combination-sum-ii/description)<br>
```python
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        # write your code here
        if not num:
            return []
        num.sort()
        self.res = []
        self.use = [0] * len(num)
        self.dfs(num, target, 0, [])
        return self.res
    def dfs(self, candidates, target, start, combination):
        if target == 0:
            self.res.append(list(combination))
            return
        for i in range(start, len(candidates)):
            if candidates[i] <= target and (i == 0 or candidates[i-1] != candidates[i] or self.use[i-1] == 1):
                combination.append(candidates[i])
                self.use[i] = 1
                self.dfs(candidates, target-candidates[i], i+1, combination)
                combination.pop()
                self.use[i] = 0
```

### 分割回文串

[分割回文串](https://www.lintcode.com/problem/palindrome-partitioning/description)<br>
```python
class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """
    def partition(self, s):
        # write your code here
        if not s or len(s) == 0:
            return []
        self.res = []
        self.dfs(s, [])
        return self.res
    def dfs(self, s, stringlist):
        if len(s) == 0:
            self.res.append(list(stringlist))
            return
        for i in range(1, len(s)+1):
            prefix = s[:i]
            if self.is_palindrome(prefix):
                self.dfs(s[i:], stringlist+[prefix])
    def is_palindrome(self, s):
        return s == s[::-1]
```

### 全排列

[全排列](https://www.lintcode.com/problem/permutations/description)<br>
```python
class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    def permute(self, nums):
        # write your code here
        if not nums:
            return [[]]
        self.res = []
        self.visited = [False] * len(nums) # 初始化为False
        self.dfs(nums, [])
        return self.res
    def dfs(self, nums, permutation):
        if len(nums) == len(permutation):
            self.res.append(list(permutation))
            return
        for i in range(len(nums)):
            if self.visited[i]:
                continue
            permutation.append(nums[i])
            self.visited[i] = True
            self.dfs(nums, permutation)
            self.visited[i] = False
            permutation.pop()
```

### 全排列2

[全排列 II(带重复元素的排列)](https://www.lintcode.com/problem/permutations-ii/description)<br>
```python
class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        # write your code here
        if not nums:
            return [[]]
        nums.sort()
        self.res = []
        self.visited = [False] * len(nums)
        self.dfs(nums, [])
        return self.res
    def dfs(self, nums, permutation):
        if len(nums) == len(permutation):
            self.res.append(list(permutation))
            return
        for i in range(len(nums)):
            if self.visited[i]:
                continue
            if i > 0 and nums[i] == nums[i-1] and (not self.visited[i-1]):
                continue
            permutation.append(nums[i])
            self.visited[i] = True
            self.dfs(nums, permutation)
            permutation.pop()
            self.visited[i] = False
```

### n皇后问题

[N皇后问题](https://www.lintcode.com/problem/n-queens/description)<br>
```python
class Solution:
    """
    @param: n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n):
        # write your code here
        if n < 1:
            return []
        self.res = []
        self.search(n, [])
        return self.res
    def search(self, n, cols):
        if len(cols) == n:
            self.res.append(self.drawchessboard(cols))
            return
        for col_index in range(n):
            if not self.is_valid(cols, col_index):
                continue
            cols.append(col_index)
            self.search(n, cols)
            cols.pop()
    def is_valid(self, cols, col_index):
        row = len(cols)
        for row_idx in range(row):
            if cols[row_idx] == col_index:
                return False
            if row_idx + cols[row_idx] == row + col_index:
                return False
            if row_idx - cols[row_idx] == row - col_index:
                return False
        return True
    def drawchessboard(self, cols):
        draw_res = []
        for i in cols:
            s = ['.'] * len(cols)
            s[i] = 'Q'
            draw_res.append(''.join(s))
        return draw_res
```

### 字符串解码

[字符串解码]()<br>
```python
# 递归方法
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def expressionExpand(self, s):
        # write your code here
        if not s or len(s) == 0:
            return ''
        res, pos = self.dfs(s, 0)
        return res
    def dfs(self, s, index):
        res = ''
        number = 0
        while index < len(s):
            if s[index].isalpha():
                res += s[index]
                index += 1
            elif s[index].isdigit():
                number = 10 * number + int(s[index])
                index += 1
            elif s[index] == '[':
                substring, pos = self.dfs(s, index + 1)
                res += number * substring
                number = 0
                index = pos
            else:
                index += 1
                return res, index
        return res, index
```

### 摊平嵌套的列表

[摊平嵌套的列表](https://www.lintcode.com/problem/flatten-nested-list-iterator/description)<br>

摊平嵌套的列表(直接摊平，和题目无关)<br>
```python
def flatten_any_list(l):
    # 先把列表变成str
    l = str(l)
    # 把str中的数字提取出来
    index = 0
    res = []
    while index < len(l):
        if l[index].isdigit():
            start = index
            index += 1
            while index < len(l) and l[index].isdigit():
                index += 1
            res.append(int(l[start:index]))
        else:
            index += 1
    return res
```

### 用栈实现队列

[用栈实现队列](https://www.lintcode.com/problem/implement-queue-by-two-stacks/description)<br>
```python
from collections import deque
class MyQueue:
    
    def __init__(self):
        # do intialization if necessary
        self.stack1 = deque()
        self.stack2 = deque()

    """
    @param: element: An integer
    @return: nothing
    """
    def push(self, element):
        # write your code here
        self.stack2.append(element)

    """
    @return: An integer
    """
    def pop(self):
        # write your code here
        if len(self.stack1) == 0:
            self.stack2Tostack1()
        return self.stack1.pop()

    """
    @return: An integer
    """
    def top(self):
        # write your code here
        if len(self.stack1) == 0:
            self.stack2Tostack1()
        return self.stack1[-1]
    # new func
    def stack2Tostack1(self):
        while self.stack2:
            self.stack1.append(self.stack2.pop())
```

### 直方图最大矩形覆盖

[直方图最大矩形覆盖](https://www.lintcode.com/problem/largest-rectangle-in-histogram/description)<br>
```python
# 自己写的，会出现Time Limit Exceeded
class Solution:
    """
    @param height: A list of integer
    @return: The area of largest rectangle in the histogram
    """
    def largestRectangleArea(self, height):
        # write your code here
        if not height:
            return 0
        max_res = 0
        for current_idx in range(len(height)):
            left = current_idx
            right = current_idx
            while left > 0 and height[current_idx] <= height[left-1]:
                left -= 1
            while right < len(height) - 1 and height[current_idx] <= height[right+1]:
                right += 1
            print(left, right)
            max_res = max(max_res, (right-left+1)*height[current_idx])
        return max_res
```

```python
# 九章算法给出的答案 stack解决
class Solution:
    """
    @param height: A list of integer
    @return: The area of largest rectangle in the histogram
    """
    def largestRectangleArea(self, height):
        # write your code here
        if not height:
            return 0
        stack = []
        max_res = 0
        for i in range(len(height)+1):
            curt = -1 if i == len(height) else height[i]
            while len(stack) != 0 and (curt <= height[stack[-1]]):
                h = height[stack.pop()]
                w = i if len(stack) == 0 else i - stack[-1] -1
                max_res = max(max_res, h * w)
            stack.append(i)
        return max_res
```

### 带最小值操作的栈

[带最小值操作的栈 min-stack](https://www.lintcode.com/problem/min-stack/description)<br>
```python
class MinStack:
    
    def __init__(self):
        # do intialization if necessary
        self.stack = []
        self.min_stack = []

    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, number):
        # write your code here
        self.stack.append(number)
        if len(self.min_stack) == 0:
            self.min_stack.append(number)
        else:
            self.min_stack.append(min(number, self.min_stack[-1]))

    """
    @return: An integer
    """
    def pop(self):
        # write your code here
        self.min_stack.pop()
        return self.stack.pop()

    """
    @return: An integer
    """
    def min(self):
        # write your code here
        return self.min_stack[-1]
```

---

### 翻转链表

[翻转链表](https://www.lintcode.com/problem/reverse-linked-list/description)<br>
```python
"""
Definition of ListNode

class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head: n
    @return: The new head of reversed linked list.
    """
    def reverse(self, head):
        # write your code here
        prev = None
        while head:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        return prev
```

### 翻转链表2

[翻转链表 II](https://www.lintcode.com/problem/reverse-linked-list-ii/description)<br>
```python
class Solution:
    """
    定义dummy node
    把m前一个和n后一个标记一下
    把m和n之间的链表反转一下
    //connect
    return dummy.next
    """
    def reverseBetween(self, head, m, n):
        # write your code here
        dummy = ListNode(0)
        dummy.next = head
        head = dummy
        for _ in range(m-1):
            head = head.next
        mprev = head
        mcurrent = head.next
        for _ in range(m-1,n):
            head = head.next
        ncurrent = head
        nplus = head.next
        
        prev = None
        curt = mcurrent
        while curt != nplus:
            tmp = curt.next
            curt.next = prev
            prev = curt
            curt = tmp
        mprev.next = ncurrent
        mcurrent.next = nplus
        return dummy.next
```

### K组翻转链表

[K组翻转链表](https://www.lintcode.com/problem/reverse-nodes-in-k-group/description)<br>
```python
class Solution:
    """
    @param head: a ListNode
    @param k: An integer
    @return: a ListNode
    """
    def reverseKGroup(self, head, k):
        # write your code here
        dummy = ListNode(0)
        dummy.next = head
        
        head = dummy
        while True:
            head = self.reverseK(head, k)
            if not head:
                break
        return dummy.next
    def reverseK(self, head, k):
        nk = head
        for _ in range(k):
            if not nk:
                return None
            nk = nk.next
        if not nk:
            return None
        # reverse
        n1 = head.next
        nkplus = nk.next
        
        # 这一步做的就是把n1和nk指针反转过来
        prev = None
        curt = n1
        while curt != nkplus:
            tmp = curt.next
            curt.next = prev
            prev = curt
            curt = tmp
        
        # connect
        head.next = nk
        n1.next = nkplus
        return n1
```

### 链表划分

[链表划分](https://www.lintcode.com/problem/partition-list/description)<br>
```python
class Solution:
    """
    @param head: The first node of linked list
    @param x: An integer
    @return: A ListNode
    """
    def partition(self, head, x):
        # write your code here
        if not head:
            return None
        leftdummy = ListNode(0)
        rightdummy = ListNode(0)
        left = leftdummy
        right = rightdummy
        while head:
            if head.val < x:
                left.next = head
                left = head
            else:
                right.next = head
                right = head
            head = head.next
        right.next = None
        left.next = rightdummy.next
        return leftdummy.next
```

### 合并两个排序链表

[合并两个排序链表](https://www.lintcode.com/problem/merge-two-sorted-lists/description)<br>
```python
class Solution:
    """
    @param l1: ListNode l1 is the head of the linked list
    @param l2: ListNode l2 is the head of the linked list
    @return: ListNode head of linked list
    """
    def mergeTwoLists(self, l1, l2):
        # write your code here
        dummy = ListNode(0)
        head = dummy
        while l1 and l2:
            if l1.val < l2.val:
                head.next = l1
                l1 = l1.next
            else:
                head.next = l2
                l2 = l2.next
            head = head.next
        if l1:
            head.next = l1
        if l2:
            head.next = l2
        return dummy.next
```

### 交换链表当中两个节点

[交换链表当中两个节点](https://www.lintcode.com/problem/swap-two-nodes-in-linked-list/description)<br>
```python
class Solution:
    """
    @param head: a ListNode
    @param v1: An integer
    @param v2: An integer
    @return: a new head of singly-linked list
    """
    def swapNodes(self, head, v1, v2):
        # write your code here
        dummy = ListNode(0)
        dummy.next = head
        head = dummy
        v1_curt = None
        v2_curt = None
        while head.next:
            if head.next.val == v1:
                v1_prev = head
                v1_curt = head.next
                v1_post = head.next.next
            if head.next.val == v2:
                v2_prev = head
                v2_curt = head.next
                v2_post = head.next.next
            head = head.next
        if (not v1_curt) or (not v2_curt):
            return dummy.next
        # connect
        # v1，v2相邻 v1 -> v2
        if v1_curt.next == v2_curt:
            v1_prev.next = v2_curt
            v2_curt.next = v1_curt
            v1_curt.next = v2_post
        # v1，v2相邻 v2 -> v1
        elif v2_curt.next == v1_curt:
            v2_prev.next = v1_curt
            v1_curt.next = v2_curt
            v2_curt.next = v1_post
        # v1，v2不相邻
        else:
            v1_prev.next = v2_curt
            v2_curt.next = v1_post
            v2_prev.next = v1_curt
            v1_curt.next = v2_post
        return dummy.next
```

### 重排链表

[重排链表](https://www.lintcode.com/problem/reorder-list/description)<br>
```python
class Solution:
    """
    @param head: The head of linked list.
    @return: nothing
    """
    def reorderList(self, head):
        # write your code here
        if head == None:
            return None
        mid = self.findMiddle(head)
        tail = self.reverse(mid.next)
        mid.next = None
        self.merge(head, tail)
    def findMiddle(self, head):
        slow = head
        fast = head.next
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
        return slow
    def reverse(self, head):
        prev = None
        while head != None:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        return prev
    def merge(self, head1, head2):
        index = 0
        dummy = ListNode(0)
        while head1 != None and head2 != None:
            if index % 2 == 0:
                dummy.next = head1
                head1 = head1.next
            else:
                dummy.next = head2
                head2 = head2.next
            dummy = dummy.next
            index += 1
        if head1 != None:
            dummy.next = head1
        if head2 != None:
            dummy.next = head2
```

### 旋转链表

[旋转链表](https://www.lintcode.com/problem/rotate-list/description)<br>
```python
class Solution:
    """
    @param head: the List
    @param k: rotate to the right k places
    @return: the list after rotation
    """
    def rotateRight(self, head, k):
        # write your code here
        if head == None or k == 0:
            return head
        curnode = head
        size = 0
        while curnode != None:
            size += 1
            curnode = curnode.next
        k = k % size
        if k == 0:
            return head
        index = 1
        curnode = head
        while index < size - k:
            curnode = curnode.next
            index += 1
        newHead = curnode.next
        curnode.next = None
        dummy = ListNode(0)
        dummy.next = newHead
        while newHead.next != None:
            newHead = newHead.next
        newHead.next = head
        return dummy.next
```

### 带环链表

[带环链表](https://www.lintcode.com/problem/linked-list-cycle/description)<br>
```python
class Solution:
    """
    @param head: The first node of linked list.
    @return: True if it has a cycle, or false
    """
    def hasCycle(self, head):
        # write your code here
        if head == None or head.next == None:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if fast == None or fast.next == None:
                return False
            fast = fast.next.next
            slow = slow.next
        return True
```

### 带环链表2

[带环链表 II](https://www.lintcode.com/problem/linked-list-cycle-ii/description)<br>
```python
class Solution:
    """
    @param head: The first node of linked list.
    @return: The node where the cycle begins. if there is no cycle, return null
    """
    def detectCycle(self, head):
        # write your code here
        if head == None or head.next == None:
            return None
        slow = head
        fast = head.next
        while fast != slow:
            if fast == None or fast.next == None:
                return None
            fast = fast.next.next
            slow = slow.next
        # 找入口
        while head != slow.next:
            head = head.next
            slow = slow.next
        return head
```

### 最大子数组

[最大子数组](https://www.lintcode.com/problem/maximum-subarray/description)<br>
```python
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        if not nums:
            return 0
        max_res = -sys.maxsize
        prefix_sum = 0
        min_sum = 0
        for i in nums:
            prefix_sum += i
            max_res = max(max_res, prefix_sum - min_sum)
            min_sum = min(prefix_sum, min_sum)
        return max_res
```

### 子数组之和

[子数组之和](https://www.lintcode.com/problem/subarray-sum/description)<br>
```python
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return []
        use_map = {0:-1}
        prefix_sum = 0
        for i in range(len(nums)):
            prefix_sum += nums[i]
            if prefix_sum in use_map:
                return [use_map[prefix_sum]+1, i]
            use_map[prefix_sum] = i
        return []
```

### 最接近零的子数组和

[最接近零的子数组和](https://www.lintcode.com/problem/subarray-sum-closest/description)<br>
```python
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySumClosest(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return []
        prefix_sum = [(0,-1)]
        for i, num in enumerate(nums):
            prefix_sum.append((prefix_sum[-1][0]+num, i))
        
        prefix_sum = sorted(prefix_sum, key=lambda x : x[0])
        
        closest = sys.maxsize
        res = []
        for i in range(1, len(prefix_sum)):
            if closest > prefix_sum[i][0] - prefix_sum[i-1][0]:
                closest = prefix_sum[i][0] - prefix_sum[i-1][0]
                left = min(prefix_sum[i][1], prefix_sum[i-1][1]) + 1
                right = max(prefix_sum[i][1], prefix_sum[i-1][1])
                res = [left, right]
        return res
```

### 整数排序2

[整数排序2](https://www.lintcode.com/problem/sort-integers-ii/description)<br>
```python
# 快速排序 quick sort
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self, A):
        # write your code here
        self.quick_sort(A, 0, len(A)-1)
    def quick_sort(self, A, left, right):
        if not A or len(A) <= 1 or left >= right:
            return A
        start = left
        end = right
        pivot = A[left]
        while left < right:
            while left < right and A[right] >= pivot:
                right -= 1
            A[left], A[right] = A[right], A[left]
            while left < right and A[left] <= pivot:
                left += 1
            A[left], A[right] = A[right], A[left]
        self.quick_sort(A, start, left-1)
        self.quick_sort(A, right+1, end)
```
```python
# 合并排序 merge sort
# 不会覆盖原来的arr
def merge_sort(arr):
    # 归并排序
    if len(arr) <= 1:
        return arr
    num = len(arr) >> 1
    left = merge_sort(arr[:num])
    right = merge_sort(arr[num:])
    return merge(left, right)

def merge(left, right):
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    ## 左边或右边list走完了，肯定会有一方剩下的直接append就行，肯定是左或右同一组最大的几个剩下
    # result += left[i:] if len(right[j:]) == 0 else right[j:]
    return result
```

### 合并排序数组

[合并排序数组](https://www.lintcode.com/problem/merge-sorted-array/description)<br>
```python
class Solution:
    def mergeSortedArray(self, A, m, B, n):
        # write your code here
        i = m - 1
        j = n - 1
        index = m + n - 1
        while i >= 0 and j >= 0:
            if A[i] > B[j]:
                A[index] = A[i]
                index -= 1
                i -= 1
            else:
                A[index] = B[j]
                index -= 1
                j -= 1
        while i >= 0:
            A[index] = A[i]
            index -= 1
            i -= 1
        while j >= 0:
            A[index] = B[j]
            index -= 1
            j -= 1
```

### 合并排序数组2

[合并排序数组 II](https://www.lintcode.com/problem/merge-two-sorted-arrays/description)<br>
```python
class Solution:
    def mergeSortedArray(self, A, B):
        # write your code here
        if not A:
            return B
        if not B:
            return A
        res = []
        i, j = 0, 0
        while i < len(A) and j < len(B):
            if A[i] <= B[j]:
                res.append(A[i])
                i += 1
            else:
                res.append(B[j])
                j += 1
        res += A[i:]
        res += B[j:]
        return res
```

### 两个排序数组的中位数

[两个排序数组的中位数](https://www.lintcode.com/problem/median-of-two-sorted-arrays/description)<br>
```python
class Solution:
    def findMedianSortedArrays(self, A, B):
        # write your code here
        n = len(A) + len(B)
        if n % 2 == 0:
            return (self.findKth(A, B, n//2) + self.findKth(A, B, n//2+1))/2.0
        return self.findKth(A, B, n//2+1)
    def findKth(self, A, B, k):
        if len(A) == 0:
            return B[k-1]
        if len(B) == 0:
            return A[k-1]
        start = min(A[0], B[0])
        end = max(A[len(A)-1], B[len(B)-1])
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self.countSmallerOrEqual(A, mid) + self.countSmallerOrEqual(B, mid) < k:
                start = mid
            else:
                end = mid
        if self.countSmallerOrEqual(A, start) + self.countSmallerOrEqual(B, start) >= k:
            return start
        return end
    def countSmallerOrEqual(self, arr, number):
        start = 0
        end = len(arr) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if arr[mid] <= number:
                start = mid
            else:
                end = mid
        if arr[start] > number:
            return start
        if arr[end] > number:
            return end
        return len(arr)
```

### 买卖股票的最佳时机

[买卖股票的最佳时机](https://www.lintcode.com/problem/best-time-to-buy-and-sell-stock/description)<br>
```python
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        if not prices or len(prices) == 0:
            return 0
        min_val = sys.maxsize
        profit = 0
        for i in prices:
            min_val = i if i < min_val else min_val
            profit = i - min_val if (i-min_val) > profit else profit
        return profit
```

### 买卖股票的最佳时机2

[买卖股票的最佳时机 II](https://www.lintcode.com/problem/best-time-to-buy-and-sell-stock-ii/description)<br>
```python
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        if not prices or len(prices) == 0:
            return 0
        profit = 0
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            if diff > 0:
                profit += diff
        return profit
```

### 最长连续序列

[最长连续序列](https://www.lintcode.com/problem/longest-consecutive-sequence/description)<br>
```python
class Solution:
    """
    @param num: A list of integers
    @return: An integer
    """
    def longestConsecutive(self, num):
        # write your code here
        if not num or len(num) == 0:
            return 0
        num_set = set(num)
        res = 0
        for i in num:
            down = i - 1
            while down in num_set:
                num_set.remove(down)
                down -= 1
            up = i + 1
            while up in num_set:
                num_set.remove(up)
                up += 1
            res = max(res, up-down-1)
        return res
```

---

### 移动零

[移动零](https://www.lintcode.com/problem/move-zeroes/description)<br>
```python
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        # write your code here
        if not nums or len(nums) <= 1:
            return nums
        i = 0
        j = 1
        while j < len(nums):
            while j < len(nums) and nums[j] == 0:
                j += 1
            while i < j and nums[i] != 0:
                i += 1
            if i < j and j < len(nums):
                nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j += 1
        return
```

### 去除重复元素

[去除重复元素](https://www.lintcode.com/problem/remove-duplicate-numbers-in-array/description)<br>
```python
class Solution:
    def deduplication(self, nums):
        # write your code here
        if not nums or len(nums) == 0:
            return 0
        d_ = {}
        res = 0
        for i in nums:
            if i not in d_:
                d_[i] = True
                nums[res] = i
                res += 1
        return res
```

### 有效回文串

[有效回文串](https://www.lintcode.com/problem/valid-palindrome/description)<br>
```python
class Solution:
    def isPalindrome(self, s):
        # write your code here
        l = [x for x in list(s) if x.isdigit() or x.isalpha()]
        s_ = ''.join(l).lower()
        return s_ == s_[::-1]
```

### 数组划分

[数组划分](https://www.lintcode.com/problem/partition-array/description)<br>
```python
class Solution:
    def partitionArray(self, nums, k):
        # write your code here
        if not nums or len(nums) <= 1:
            return 0
        i = 0
        j = len(nums) - 1
        while i < j:
            while i < j and nums[i] < k:
                i += 1
            while i < j and nums[j] >= k:
                j -= 1
            if i < j:
                nums[i], nums[j] = nums[j], nums[i]
        if nums[i] < k:
            return i + 1
        return i
```

### 无序数组K小元素

[无序数组K小元素](https://www.lintcode.com/problem/kth-smallest-numbers-in-unsorted-array/description)<br>
```python
class Solution:
    """
    @param k: An integer
    @param nums: An integer array
    @return: kth smallest element
    """
    def kthSmallest(self, k, nums):
        # write your code here
        return self.quickSelect(nums, 0, len(nums)-1, k-1)
    def quickSelect(self, A, start, end, k):
        if start == end:
            return A[start]
        left = start
        right = end
        mid = start + (end - start) // 2
        pivot = A[mid]
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            while left <= right and A[right] > pivot:
                right -= 1
            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
        if right >= k and right >= start:
            return self.quickSelect(A, start, right, k)
        elif left <= k and left <= end:
            return self.quickSelect(A, left, end, k)
        else:
            return A[k]
```

### 交错正负数

[交错正负数](https://www.lintcode.com/problem/interleaving-positive-and-negative-numbers/description)<br>
```python
class Solution:

    def rerange(self, A):
        # write your code here
        if not A or len(A) == 0:
            return 
        A.sort()
        
        A_pos = len([x for x in A if x > 0])
        A_neg = len([x for x in A if x < 0])
        
        if A_neg > A_pos:
            i = 1
            j = len(A) - 1
        elif A_neg == A_pos:
            i = 0
            j = len(A) - 1
        else:
            i = 0
            j = len(A) - 2
        idx = 0
        while i < j:
            if idx % 2 == 0:
                A[i], A[j] = A[j], A[i]
            i += 1
            j -= 1
            idx += 1
        return
```

### 字符大小写排序

[字符大小写排序](https://www.lintcode.com/problem/sort-letters-by-case/description)<br>
```python
class Solution:
    """
    @param: chars: The letter array you should sort by Case
    @return: nothing
    """
    def sortLetters(self, chars):
        # write your code here
        if not chars or len(chars) <= 1:
            return chars
        chars.sort(key=lambda x : x.isupper())
```

### 两数之和

[两数之和](https://www.lintcode.com/problem/two-sum/description)<br>
```python
# 自己想的
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        # write your code here
        if not numbers or len(numbers) == 0:
            return None
        nums = [(j, i) for i, j in enumerate(numbers)]
        nums.sort(key=lambda x : x[0])
        i = 0
        j = len(numbers) - 1
        while i < j:
            if nums[i][0] + nums[j][0] == target:
                start = min(nums[i][1], nums[j][1])
                end = max(nums[i][1], nums[j][1])
                return [start, end]
            elif nums[i][0] + nums[j][0] < target:
                i += 1
            else:
                j -= 1
        return None
```

### 两数之和_不同组成

[两数之和 - 不同组成](https://www.lintcode.com/problem/two-sum-unique-pairs/description)<br>
```python
class Solution:

    def twoSum6(self, nums, target):
        # write your code here
        if not nums or len(nums) == 0:
            return 0
        nums.sort()
        i = 0
        j = len(nums) - 1
        res = 0
        uniq_lst = set()
        while i < j:
            if nums[i] + nums[j] == target:
                tmp = str(nums[i]) + str(nums[j])
                if tmp not in uniq_lst:
                    res += 1
                    uniq_lst.add(tmp)
                i += 1
                j -= 1
            elif nums[i] + nums[j] < target:
                i += 1
            else:
                j -= 1
        return res
```

### 三数之和

[三数之和](https://www.lintcode.com/problem/3sum/description)<br>
```python
# 思路：先固定一个指针，另外两个移动
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        # write your code here
        if not numbers or len(numbers) <= 2:
            return []
        numbers.sort()
        res = []
        for i in range(len(numbers)-2):
            if i != 0 and (numbers[i] == numbers[i-1]):
                continue
            j = i + 1
            k = len(numbers) - 1
            while j < k:
                if numbers[i] + numbers[j] + numbers[k] == 0:
                    res.append((numbers[i], numbers[j], numbers[k]))
                    j += 1
                    k -= 1
                    while j < k and numbers[j] == numbers[j-1]:
                        j += 1
                    while j < k and numbers[k] == numbers[k+1]:
                        k -= 1
                elif numbers[i] + numbers[j] + numbers[k] < 0:
                    j += 1
                else:
                    k -= 1
        return res
```

### 两数和_小于或等于目标值

[两数和-小于或等于目标值](https://www.lintcode.com/problem/two-sum-less-than-or-equal-to-target/leaderboard)<br>
```python
class Solution:
    """
    @param nums: an array of integer
    @param target: an integer
    @return: an integer
    """
    def twoSum5(self, nums, target):
        # write your code here
        if not nums or len(nums) <= 1:
            return 0
        nums.sort()
        res = 0
        i = 0
        j = len(nums) - 1
        while i < j:
            if nums[i] + nums[j] <= target:
                res += (j - i)
                i += 1
            else:
                j -= 1
        return res
```

### 最接近的三数之和

[最接近的三数之和](https://www.lintcode.com/problem/3sum-closest/description)<br>
```python
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """
    def threeSumClosest(self, numbers, target):
        # write your code here
        if not numbers or len(numbers) <= 2:
            return None
        numbers.sort()
        res = numbers[0] + numbers[1] + numbers[2]
        for i in range(len(numbers) - 2):
            j = i + 1
            k = len(numbers) - 1
            while j < k:
                s = numbers[i] + numbers[j] + numbers[k]
                if abs(target - res) > abs(target - s):
                    res = s
                elif s < target:
                    j += 1
                else:
                    k -= 1
        return res
```

### 四数之和

[四数之和](https://www.lintcode.com/problem/4sum/description)<br>
```python
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, numbers, target):
        # write your code here
        if not numbers or len(numbers) <= 3:
            return []
        numbers.sort()
        res = []
        for i in range(len(numbers)-3):
            if i != 0 and numbers[i] == numbers[i-1]:
                continue
            for j in range(i+1, len(numbers)-2):
                if j != i + 1 and numbers[j] == numbers[j-1]:
                    continue
                k = j + 1
                l = len(numbers) - 1
                new_target = target - numbers[i] - numbers[j]
                while k < l:
                    if numbers[k] + numbers[l] == new_target:
                        res.append((numbers[i],numbers[j],numbers[k],numbers[l]))
                        k += 1
                        l -= 1
                        while k < l and numbers[k] == numbers[k-1]:
                            k += 1
                        while k < l and numbers[l] == numbers[l+1]:
                            l -= 1
                    elif numbers[k] + numbers[l] < new_target:
                        k += 1
                    else:
                        l -= 1
        return res
```

### 两数和_差等于目标值

[两数和 - 差等于目标值](https://www.lintcode.com/problem/two-sum-difference-equals-to-target/description)<br>
```python
# 参考别人的
class Solution:
    """
    @param nums: an array of Integer
    @param target: an integer
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum7(self, nums, target):
        # Write your code here
        nums = [(num, i) for i, num in enumerate(nums)]
        target = abs(target)    
        n, indexs = len(nums), []
    
        nums = sorted(nums, key=lambda x: x[0])

        j = 0
        for i in range(n-1):
            if i == j:
                j += 1
            while j < n and nums[j][0] - nums[i][0] < target:
                j += 1
            if j < n and nums[j][0] - nums[i][0] == target:
                indexs = [nums[i][1] + 1, nums[j][1] + 1]

        if indexs[0] > indexs[1]:
            indexs[0], indexs[1] = indexs[1], indexs[0]

        return indexs
```

---

### 用栈实现队列

[用栈实现队列](https://www.lintcode.com/problem/implement-queue-by-two-stacks/description)

### 双队列实现栈

[双队列实现栈](https://www.lintcode.com/problem/implement-stack-by-two-queues/description)

### 重哈希

[重哈希](https://www.lintcode.com/problem/rehashing/description)<br>
```python
class Solution:
    """
    @param hashTable: A list of The first node of linked list
    @return: A list of The first node of linked list which have twice size
    """
    def rehashing(self, hashTable):
        # write your code here
        hash_size = 2 * len(hashTable)
        rehashTable = [None for i in range(hash_size)]
        for item in hashTable:
            p = item
            while p != None:
                self.addnode(rehashTable, p.val)
                p = p.next
        return rehashTable
    def addnode(self, rehashTable, val):
        p = val % len(rehashTable)
        if rehashTable[p] == None:
            rehashTable[p] = ListNode(val)
        else:
            self.addnodelist(rehashTable[p], val)
    def addnodelist(self, node, val):
        if node.next != None:
            self.addnodelist(node.next, val)
        else:
            node.next = ListNode(val)
```

### LRU缓存策略

[LRU缓存策略](https://www.lintcode.com/problem/lru-cache/description)<br>
```python
# 注：key指向LinkedList当前节点的前一个prev
class LinkedNode:
    def __init__(self, key=None, val=None, next=None):
        self.key = key
        self.val = val
        self.next = next

class LRUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        # do intialization if necessary
        self.hash = {}
        self.head = LinkedNode()
        self.tail = self.head
        self.capacity = capacity

    """
    @param: key: An integer
    @return: An integer
    """
    def get(self, key):
        # write your code here
        if key not in self.hash:
            return -1
        self.kick(self.hash[key])
        return self.hash[key].next.val

    """
    @param: key: An integer
    @param: value: An integer
    @return: nothing
    """
    def set(self, key, value):
        # write your code here
        if key in self.hash:
            self.kick(self.hash[key])
            self.hash[key].next.val = value
        else:
            self.push_back(LinkedNode(key, value))
            if len(self.hash) > self.capacity:
                self.pop_front()
    def push_back(self, node):
        self.hash[node.key] = self.tail
        self.tail.next = node
        self.tail = node
    def pop_front(self):
        del self.hash[self.head.next.key]
        self.head.next = self.head.next.next
        self.hash[self.head.next.key] = self.head
    def kick(self, prev):
        node = prev.next
        if node == self.tail:
            return
        prev.next = node.next
        if node.next is not None:
            self.hash[node.next.key] = prev
            node.next = None
        self.push_back(node)
```

### 乱序字符串

[乱序字符串](https://www.lintcode.com/problem/anagrams/description)<br>
```python
class Solution:
    """
    @param strs: A list of strings
    @return: A list of strings
    """
    def anagrams(self, strs):
        # write your code here
        d_ = {}
        res = []
        for s in strs:
            s_reorder = ''.join(sorted(s))
            if s_reorder in d_:
                d_[s_reorder].append(s)
            else:
                d_[s_reorder] = [s]
        for k, v in d_.items():
            if len(v) > 1:
                res += v
        return res
```

### 堆化

[堆化 Heapify](https://www.lintcode.com/problem/heapify/description)<br>
对于每个元素A[i]，比较A[i]和它的父亲结点的大小，如果小于父亲结点，则与父亲结点交换。
交换后再和新的父亲比较，重复上述操作，直至该点的值大于父亲。
对于每个元素都要遍历一遍，这部分是 O(n)。每处理一个元素时，最多需要向根部方向交换 logn 次。因此总的时间复杂度是 O(nlogn)。
<br>
```python
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # write your code here
        if not A or len(A) == 0:
            return
        for i in range(len(A)):
            self.siftup(A, i)
    def siftup(self, A, k):
        while k != 0:
            father = (k - 1) // 2
            if A[k] > A[father]:
                break
            A[father], A[k] = A[k], A[father]
            k = father
```

### 丑数2

[丑数 II](https://www.lintcode.com/problem/ugly-number-ii/description)<br>
```python
class Solution:
    """
    @param n: An integer
    @return: the nth prime number as description.
    """
    def nthUglyNumber(self, n):
        # write your code here
        if n <= 0:
            return None
        if n == 1:
            return 1
        uglys = [1]
        p2, p3, p5 = 0, 0, 0
        for i in range(1, n):
            lastNumber = uglys[-1]
            while uglys[p2] * 2 <= lastNumber:
                p2 += 1
            while uglys[p3] * 3 <= lastNumber:
                p3 += 1
            while uglys[p5] * 5 <= lastNumber:
                p5 += 1
            uglys.append(min(uglys[p2]*2, uglys[p3]*3, uglys[p5]*5))
        return uglys[-1]
```

### 最高频的k个单词

[最高频的K个单词](https://www.lintcode.com/problem/top-k-frequent-words/description)<br>
```python
from collections import Counter
class Solution:
    """
    @param words: an array of string
    @param k: An integer
    @return: an array of string
    """
    def topKFrequentWords(self, words, k):
        # write your code here
        word_count = Counter(words)
        # 这里用sys.maxsize是为了反转数字，从小到大排列，和字母一直顺序。如果有个数大于sys.maxsize则会有问题
        word_count_order = sorted(word_count.items(),key=lambda x : str(sys.maxsize-x[1]) + x[0])
        return [x[0] for x in word_count_order][:k]
```

---

### 数字三角形

递归和动规入门题及提升<br>
[数字三角形](https://www.lintcode.com/problem/triangle/description)<br>
```python

```