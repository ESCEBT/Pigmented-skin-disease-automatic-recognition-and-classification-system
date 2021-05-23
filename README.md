项目名称：  
色素性皮肤病七分类系统（微信版）

网站版项目地址：  
https://github.com/JunhaoCheng/Pigmented-skin-disease-automatic-recognition-and-classification-system/tree/website
    
项目功能：  
基于深度学习、集成学习、迁移学习、GAN等技术的色素性皮肤病自动识别七分类系统。  
本系统主要由服务端和客户端两个模块组成。服务端使用DenseNet161和SENet154  
两个模型构成集成模型，从而实现了对色素性皮肤病自动识别七分类。客户端使  
用微信小程序和网站(SSM、Springboot)开发。用户通过微信小程序或网站上传图像到服务端，服务端返回所属类别。

项目组织结构：  
服务端：  
![Image text](https://github.com/JunhaoCheng/-Pigmented-skin-disease-automatic-recognition-and-classification-system-/blob/master/Imgs/Picture4.png)  


项目部署：  
## 项目名称：  
色素性皮肤病七分类系统（网站版）
***
## 项目差异:  
与微信版的差异主要体现在:  
服务端:  
&nbsp;&nbsp;使用Resnet152、InceptionV3、EfficientNet作集成学习，  
&nbsp;&nbsp;并使用DGGAN、WGAN_GP合成数据  
客户端:  
&nbsp;&nbsp;基于SSM和SpringBoot两个版本开发，拥有皮肤病预测、皮肤病资讯、  
&nbsp;&nbsp;病例图片查询三大功能
***
## 项目部署：  
1、修改server.py文件并运行  
2、修改client.py文件并运行(可选)  
3、修改网站客户端服务器配置
***
## 项目运行效果：  
![Image text](https://github.com/JunhaoCheng/Pigmented-skin-disease-automatic-recognition-and-classification-system/blob/website/%E8%BF%90%E8%A1%8C%E6%95%88%E6%9E%9C%E5%9B%BE%E7%89%87/Picture1.png)![Image text](https://github.com/JunhaoCheng/Pigmented-skin-disease-automatic-recognition-and-classification-system/blob/website/%E8%BF%90%E8%A1%8C%E6%95%88%E6%9E%9C%E5%9B%BE%E7%89%87/Picture2.png)![Image text](https://github.com/JunhaoCheng/Pigmented-skin-disease-automatic-recognition-and-classification-system/blob/website/%E8%BF%90%E8%A1%8C%E6%95%88%E6%9E%9C%E5%9B%BE%E7%89%87/Picture3.png)  
***
## 项目演示视屏：  
https://github.com/JunhaoCheng/Pigmented-skin-disease-automatic-recognition-and-classification-system/blob/website/%E6%BC%94%E7%A4%BA%E8%A7%86%E5%B1%8F.mov


# install packages
pip3 install pandas
pip3 install sklearn
