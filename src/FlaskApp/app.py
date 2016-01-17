from flask import Flask, render_template, request
from pyelasticsearch import ElasticSearch


es = ElasticSearch('http://localhost:9200/')
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("index2.html")

@app.route("/abc", methods=['POST'])
def disp():
    a = request.form['movie1']
    r = request.form['rating']
    #st=es.search('movieID: 2', index='moviedb')
    #print es.search(body={"query":{"match_phrase_prefix":{"moviename":"To"}}},index='moviedb')
    #query='movieID:1'
    #query={'query':{'match_phrase_prefix':{'moviename':'T'}}}
    query={'fields':['moviename','movieID'],'query':{'match_phrase_prefix':{'moviename':a}}}
    
    
    # Output of PyElasticSearch is Dictionary
    st=es.search(query, index='moviedb')
    #fields=st.split(',')
    
    count = st['hits']['total']
    for i in range(count):
    	print st['hits']['hits'][i]['fields']['movieID']

    #print es.search(body={"query":{"match_all":{}}},index='moviedb')
    #return render_template("abc.html",a=st)
    return render_template("abc.html",a=st)

if __name__ == "__main__":
    app.run()

