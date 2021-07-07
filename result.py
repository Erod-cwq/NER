import json

with open('./result.json', 'r') as f:
    results = json.load(f)

texts = []
with open('./data/final_test.txt', 'r', encoding='utf8') as f:
    for line in f.readlines():
        text = line.strip().split('')[1]
        texts.append(text)

print(len(texts))

print(len(results))


with open('./个人队_addr_parsing_runid.txt', 'w', encoding='utf8') as f:
    for index in range(len(results)):
        print(index+1)
        result = results[index]
        text = texts[index]
        assert len(result)-2 == len(text)
        f.write(str(index+1))
        f.write(u'\u0001')
        f.write(text)
        f.write(u'\u0001')
        for item in result[:-2]:
            f.write(item + ' ')
        f.write('\n')
        index += 1

