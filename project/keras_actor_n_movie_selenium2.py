from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.request import Request, urlopen

#원본이미지로 저장가능한 크롤링소스로 변경했음.
#selenium 최신버전으로 문법이 바꼈다. 바뀐걸로 적용해줌.
#Chrome 드라이버 자동으로 잡아주는게 추가됨(Service, ChromeDriverManager)
#원하는 Xpath 위치 잘 봐야함
#except 작업을 아예 빼주거나 pass로 넣어버리면 작업이 멈추거나 오류남. 걍 에러표시라도 해줌

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def crawling_img(name):
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install())) #드라이버 위치 자동으로 찾아줌
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
    elem = driver.find_element(By.NAME,"q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)

    #
    SCROLL_PAUSE_TIME = 1
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")  # 브라우저의 높이를 자바스크립트로 찾음
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")  # 브라우저 끝까지 스크롤을 내림
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_elements(By.CSS_SELECTOR,".mye4qd").click() #여러개라서 s붙음
            except:
                break
        last_height = new_height

    imgs = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
    dir = "D:/project/actor"+ "/" + name
    #끝까지 다내려서 느리다.
    
    createDirectory(dir) #폴더 생성해준다
    count = 1
    for img in imgs:
        try:
            img.click()
            time.sleep(2)
            imgUrl = driver.find_element(By.XPATH,'//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img').get_attribute("src")
            path = "D:/project/actor/" + name + "/"
            urllib.request.urlretrieve(imgUrl, path + "img" + str(count) + ".jpg")
            count = count + 1
            if count >= 501: #이미지 장수 선택 
                break
        except:
            print("저장안됨") #경로못찾으면 패~쓰~~~~~
    driver.close()
actors = ["손예진"] 
 
for actor in actors:
    crawling_img(actor)