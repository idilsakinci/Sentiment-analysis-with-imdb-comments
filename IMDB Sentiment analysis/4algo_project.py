#!/usr/bin/env python
# coding: utf-8

# # NLP (Doğal Dil İşleme) ile IMDB Verileri Duygu Analizi
# ## Duygu analizi için kullanılan doğal dil işleme teknikleri, bir metindeki kelimelerin anlamını, cümlelerin yapısını, kelime dağarcığını ve benzeri özellikleri analiz ederek, metindeki duygu durumunu belirlemeye çalışır. Bu analizler sonucunda, metnin pozitif, negatif veya nötr olduğu belirlenebilir.
# #### Bu projede 4 farklı algoritma ile duygu analizi yapılmıştır.

# ### Gerekli Kütüphanelerin Dahil Edilmesi ve Veri Setinin İncelenmesi

# In[1]:


# import işlemleri yapıldı

# Numpy kütüphanesi çok boyutlu diziler ve matrislerle çalışmak için kullanılır
import numpy as np
# Pandas kütüphanesi veri işleme ve temizleme için kullanılır
import pandas as pd
# Matplotlib verileri görselleştirmek için kullanılır
import matplotlib.pyplot as plt
# CountVectorizer metin verilerini sayısal vektörlere dönüştürmek için kullanılır
from sklearn.feature_extraction.text import CountVectorizer
# BeautifulSoup HTML taglerini veri setinden temizlemek için kullanılır
from bs4 import BeautifulSoup 
# nltl doğal dil işleme için kullanılan bir kütüphanedir ve stopwords ayıklamak için kullanılır
import nltk
# stopwords İngilizce'de the, are, is gibi belirleyici olmayan kelimeleri ayrımak için kullanılır
from nltk.corpus import stopwords
# Kelime köklerini bulmak için kullanılır
from nltk.stem.porter import PorterStemmer
# Makine öğrenimi modelinin eğitim ve test verilerini ayırmak için kullanılır
from sklearn.model_selection import train_test_split
# Naive Bayes sınıflandırması için kullanılır
from sklearn.naive_bayes import GaussianNB
# Logistic Regression sınıflandırması için kullanılır
from sklearn.linear_model import LogisticRegression
# Decision Tree sınıflandırması için kullanılır
from sklearn.tree import DecisionTreeClassifier
# Random Forest Sınıflandırması için kullanılır
from sklearn.ensemble import RandomForestClassifier
# KNN sınıflandırması için kullanılır
from sklearn.neighbors import KNeighborsClassifier
# Roc_auc_score ROC eğrisi altında kalan alanın hesaplanması için kullanılır yani analizin doğruluk oranı ölçülür
from sklearn.metrics import roc_auc_score
# Sınıflandırma performansını değerlendirmek için kullanılan bir matris oluşturmak için kullanılır
from sklearn.metrics import confusion_matrix
# re düzenli ifadeler için kullanılan bir modüldür
import re


# In[2]:


# Veri seti yüklendi
# Veri setindeki yorumların birbirinden \ ile ayrıldığı belirtildi
# "" ların görmezden gelinmesi için de quoting=3 ifadesi kullanıldı
df = pd.read_csv('NLPlabeledData.tsv',  delimiter="\t", quoting=3)


# In[3]:


# Veriler incelendi
df.head()


# In[4]:


# Veri setinin uzunluğuna bakıldı
len(df)


# In[5]:


# Veri seti içinde kaç adet review(yorum) olduğuna bakıldı
len(df["review"])


# ### Veri Setinin Temizlenmesi

# In[6]:


# Stopwords temizlemek için nltk kütüphanesinden stopwords kelime seti indirildi
# Bu işlem nltk ile yapıldı
# Stopwords the, is, are gibi kelimeler anlamına gelmektedir
nltk.download('stopwords')


# In[7]:


# İşlemi yapacak fonksiyon oluşturuldu
def process(review):
    # HTML tagleri yorumların içinden temizlendi
    # get_text ile temizlenmiş veri alındı
    review = BeautifulSoup(review).get_text()
    # Harf olmayan ifadelerin hepsi boşlukla değiştirildi. Noktalama işaretleri ve sayılar ayıklanmış oldu
    review = re.sub("[^a-zA-Z]",' ',review)
    # Tüm harfler küçük harfe çevrildi
    review = review.lower()
    # Stopwords kelimelerin temizlenme işlemi liste üzerinden yapılabildiği için kelimeler işlemden önce split ile ayırılıp
    # bir listeye dönüştürüldü
    review = review.split()
    # Kelimelerin kökleri bulundu
    ps = PorterStemmer()
    # Stopwords ayıklandı
    swords = set(stopwords.words("english"))
    # Kelime stopword değilse review içine eklendi
    review = [w for w in review if w not in swords]               
    # Split edilmiş veriler boşluk ifadesiyle birleşitirildi ve temiz bir veri seti oluşmuş oldu
    return(" ".join(review))


# In[8]:


# corpus adlı bir dizi oluşturuldu
corpus = []
# Eldeki tüm veriler yukarıdaki fonksiyona göre döngüye sokuldu ve her birine ayıklanma işlemi uygulanmış oldu
for r in range(len(df["review"])): 
    # Her 1000 review sonrası bir kontrol satırı yazdırılarak review işleminin durumu kontrol edildi
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    # Ayıklanan veri corpus dizisine eklendi
    corpus.append(process(df["review"][r]))


# ### Beg of Words 
# #### Metin tabanlı verilerin sayılara ve bag of words matrisine çevrilmesi gerekiyor. Bunun için CountVectorizer kullanılmalı.¶
# #### En çok kullanılan 2000 kelime alınacak ve her bir review içinde o kelime varsa 1, yoksa 0 değeri atanacak, oluşan matris beg of words matrisidir

# In[9]:


# Sklearn içinde bulunan countvectorizer fonksiyonu kullanılarak maximum 5000 kelimelik bag of words oluşturuldu
cv = CountVectorizer(max_features = 2000)
# Train verileri feature vektör matrisine çevirildi
X = cv.fit_transform(corpus)
# Fit işlemi için X bir diziye çevrildi
X= X.toarray()


# In[10]:


# y her yorum için verilen kullanıcı duygusunu temsil eder - 0/1
y = np.array(df["sentiment"])


# ### Train ve Test Split 

# In[11]:


# 25000 verinin %80'i train, %20'si test işlemi için ayrıldı
# İşlem sırasında, "random_state" parametresi verilerin rastgele bölünmesi için kullanılan bir değeridir. Aynı veri kümesi
# üzerinde birden fazla kez çalışıldığında verilerin farklı şekillerde bölünmesi yerine aynı şekilde bölünmesini sağlar.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# ### 1. Algoritma: Naive Bayes
# #### Naive Bayes sınıflandırıcısı, sınıflandırma yapmak için metindeki özelliklerin olasılıklarını kullanır. Özellik olarak, kelime dağılımları veya kelime sıklıkları kullanılabilir. Bu özellikler, sınıflandırıcıya her kelimenin her sınıf için olasılığını hesaplamak için gereken istatistiksel bilgiyi sağlar. Sınıflandırıcı, verilen metnin her sınıf için olasılıklarını hesaplar ve en yüksek olasılığa sahip sınıfı tahmin eder. 

# In[12]:


# GaussianNB sınıflandırıcısı oluşturuldu
classifier = GaussianNB()
# Train veri setindeki özellikler (X_train) ve hedef değişken (y_train) kullanılarak sınıflandırıcının eğitilmesi sağlandı
classifier.fit(X_train, y_train)


# In[13]:


# Eğitilmiş sınıflandırıcı kullanılarak, test veri setindeki özelliklerin (X_test) sınıflandırılması yapıldı
y_predN = classifier.predict(X_test)


# In[14]:


# Gerçek sınıflar ve tahmin edilen sınıflar arasındaki benzerlik karşılaştırılarak confusion matrix oluşturuldu
cm = confusion_matrix(y_test, y_predN)
print("Confusion martrix for Naive Bayes is :- \n")
print(cm)


# ###### Çıktıya göre:
# ##### 2116 olumlu veri doğru tahmin edilmişir.
# ##### 365 veri yanlış şekilde olumlu tahmin edilmiştir.
# ##### 818 veri yanlış şekilde olumsuz tahmin edilmiştir.
# ##### 1071 olumsuz veri doğru tahmin edilmiştir.

# In[15]:


# Sonuca göre bir doğruluk yüzdesi oluşturuldu
accuracy = roc_auc_score(y_test, y_predN)
print("Accuracy rate : % ", accuracy * 100)


# #### Naive Bayes algoritmasının bu veri setindeki doğruluk oranının %76 olduğu görüldü

# ### 2. Algoritma: Logistic Regression
# #### Model, verilen girdi özelliklerinin ağırlıklarını öğrenir ve bu ağırlıkların ağırlıklı toplamı olarak bir olasılık değeri hesaplar. Bu olasılık değeri, verinin pozitif sınıfa ait olma olasılığını temsil eder.

# In[16]:


# Logistic Regression sınıflandırıcısı oluşturuldu
classifier1 = LogisticRegression(random_state = 42)
# Train veri setindeki özellikler (X_train) ve hedef değişken (y_train) kullanılarak sınıflandırıcının eğitilmesi sağlandı
classifier1.fit(X_train, y_train)


# In[17]:


# Eğitilmiş sınıflandırıcı kullanılarak, test veri setindeki özelliklerin (X_test) sınıflandırılması yapıldı
y_predL = classifier1.predict(X_test)


# In[18]:


# Gerçek sınıflar ve tahmin edilen sınıflar arasındaki benzerlik karşılaştırılarak confusion matrix oluşturuldu
cm1 = confusion_matrix(y_test, y_predL)
print("Confusion martrix for Logistic regression is :- \n")
print(cm1)


# ###### Çıktıya göre:
# ##### 2114 olumlu veri doğru tahmin edilmişir.
# ##### 367 veri yanlış şekilde olumlu tahmin edilmiştir.
# ##### 345 veri yanlış şekilde olumsuz tahmin edilmiştir.
# ##### 2174 olumsuz veri doğru tahmin edilmiştir.

# In[32]:


# Sonuca göre bir doğruluk yüzdesi oluşturuldu
accuracy = roc_auc_score(y_test, y_predL)
print("Accuracy rate : % ", accuracy * 100)


# #### Logistic Regression algoritmasının bu veri setindeki doğruluk oranının %85 olduğu görüldü

# ### 3. Algortima: Decision Trees
# #### Karar ağacı yöntemi, verileri belirli bir özellikle ilgili olarak bölerek bir ağaç yapısı oluşturur. Bu ağaç yapısı, verileri sınıflandırmak için kullanılır. Karar ağacı yöntemi, ayrıca, verilerdeki önemli özellikleri belirlemek için de kullanılır.
# #### Duygu analizi için karar ağacı yöntemi, metin verilerindeki önemli özellikleri belirleyerek, verileri sınıflandırmak için daha etkili bir şekilde kullanılabilir.

# In[20]:


# Decision Tree sınıflandırıcısı oluşturuldu
# Criterion parametresi, ağaçların oluşturulması için kullanılan bölünme ölçüsünü belirler. "entropy" veya "gini"
# Entropy değeri, daha homojen dallara sahip ağaçların oluşturulmasını sağladığı için burada o kullanıldı
classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
# Train veri setindeki özellikler (X_train) ve hedef değişken (y_train) kullanılarak sınıflandırıcının eğitilmesi sağlandı
classifier2.fit(X_train, y_train)


# In[21]:


# Eğitilmiş sınıflandırıcı kullanılarak, test veri setindeki özelliklerin (X_test) sınıflandırılması yapıldı
y_predD = classifier2.predict(X_test)


# In[22]:


# Gerçek sınıflar ve tahmin edilen sınıflar arasındaki benzerlik karşılaştırılarak confusion matrix oluşturuldu
cm2 = confusion_matrix(y_test, y_predD)
print("Confusion martrix for Decision Tree is :- \n")
print(cm2)


# ###### Çıktıya göre:
# ##### 1726 olumlu veri doğru tahmin edilmişir.
# ##### 755 veri yanlış şekilde olumlu tahmin edilmiştir.
# ##### 731 veri yanlış şekilde olumsuz tahmin edilmiştir.
# ##### 1788 olumsuz veri doğru tahmin edilmiştir.

# In[33]:


# Sonuca göre bir doğruluk yüzdesi oluşturuldu
accuracy = roc_auc_score(y_test, y_predD)
print("Accuracy rate : % ", accuracy * 100)


# #### Decision Trees algoritmasının bu veri setindeki doğruluk oranının %70 olduğu görüldü

# ### 3.Algoritma: Random Forest
# #### Random forest, bir makine öğrenimi algoritmasıdır ve sınıflandırma veya regresyon problemlerinde kullanılabilir. Random forest, birden fazla karar ağacını (decision tree) bir araya getirerek bir topluluk (ensemble) oluşturur ve bu topluluğun çoğunluğunun oylamasıyla son sınıflandırma yapar. Bu yöntem, overfitting'e karşı daha dayanıklıdır ve daha iyi bir genelleştirme sağlar.
# #### Duygu analiznde model, önceden işlenmiş metin verileri kullanarak, her veri noktasını bir sınıfa atar ve böylece bir metnin duygu durumunu tahmin edebilir.

# In[24]:


# Random Forest sınıflandırıcısı oluşturuldu. 100 adet decision kullanıldı
classifier3 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
# Train veri setindeki özellikler (X_train) ve hedef değişken (y_train) kullanılarak sınıflandırıcının eğitilmesi sağlandı
classifier3.fit(X_train, y_train)


# In[25]:


# Eğitilmiş sınıflandırıcı kullanılarak, test veri setindeki özelliklerin (X_test) sınıflandırılması yapıldı
y_predR = classifier3.predict(X_test)


# In[26]:


# Gerçek sınıflar ve tahmin edilen sınıflar arasındaki benzerlik karşılaştırılarak confusion matrix oluşturuldu
cm3 = confusion_matrix(y_test, y_predR)
print("Confusion martrix for Random Forest is :- \n")
print(cm3)


# ###### Çıktıya göre:
# ##### 2098 olumlu veri doğru tahmin edilmişir.
# ##### 383 veri yanlış şekilde olumlu tahmin edilmiştir.
# ##### 398 veri yanlış şekilde olumsuz tahmin edilmiştir.
# ##### 2130 olumsuz veri doğru tahmin edilmiştir.

# In[35]:


# Sonuca göre bir doğruluk yüzdesi oluşturuldu
accuracy = roc_auc_score(y_test, y_predR)
print("Accuracy rate : % ", accuracy * 100)


# #### Random Forest algoritmasının bu veri setindeki doğruluk oranının %84 olduğu görüldü

# ### KNN
# #### KNN algoritması, verileri etiketlerine göre gruplara ayırır ve veri noktalarının yakınlığına göre bir etiket tahmini yapar. Algoritma, veri noktalarının birbirine ne kadar yakın olduğunu ölçmek için öklid uzaklığı kullanır. Bu öklid uzaklığına göre en yakın komşular seçilir ve sınıflandırma için kullanılır.
# #### Duygu analizi için KNN algoritması kullanıldığında, veri kümesindeki metin verileri vektörel olarak ifade edilir. Daha sonra, öklid uzaklığı kullanarak, bir metnin diğerlerine ne kadar benzediği belirlenir ve metinler sınıflandırılır.

# In[28]:


# KNN sınıflandırıcısı oluşturuldu
# İşlem için 5 komşunun kullanılacağı belirlendi
# Uzaklık ölçüsünü belirleyen metric'de öklid'e benzeyen bir uzaklık olan minkowski tanımlandı
classifier4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# Train veri setindeki özellikler (X_train) ve hedef değişken (y_train) kullanılarak sınıflandırıcının eğitilmesi sağlandı
classifier4.fit(X_train, y_train)


# In[29]:


# Eğitilmiş sınıflandırıcı kullanılarak, test veri setindeki özelliklerin (X_test) sınıflandırılması yapıldı
y_predK = classifier4.predict(X_test)


# In[30]:


# Gerçek sınıflar ve tahmin edilen sınıflar arasındaki benzerlik karşılaştırılarak confusion matrix oluşturuldu
cm4 = confusion_matrix(y_test, y_predK)
print("Confusion martrix for KNN is :- \n")
print(cm4)


# ###### Çıktıya göre:
# ##### 1679 olumlu veri doğru tahmin edilmişir.
# ##### 802 veri yanlış şekilde olumlu tahmin edilmiştir.
# ##### 1090 veri yanlış şekilde olumsuz tahmin edilmiştir.
# ##### 1429 olumsuz veri doğru tahmin edilmiştir.

# In[36]:


# Sonuca göre bir doğruluk yüzdesi oluşturuldu
accuracy = roc_auc_score(y_test, y_predK)
print("Accuracy rate : % ", accuracy * 100)


# #### KNN algoritmasının bu veri setindeki doğruluk oranının %62 olduğu görüldü

# #### İncelenen dört algoritma arasından bu veri seti için en doğru sonucu logistic regression algoritmasının verdiği anlaşıldı

# In[ ]:




