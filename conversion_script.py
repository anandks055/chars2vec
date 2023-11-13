padas=['tapaḥ svādhyāya niratām tapasvī vāgvidām varam', 'nāradam paripapraccha vālmīkiḥ muni puṃgavam',
'kaḥ nu asmin sāṃpratam loke guṇavān kaḥ ca vīryavān', 'dharmajñaḥ ca kṛtajñaḥ ca satya vākyo dhṛḍha vrataḥ',
'cāritreṇa ca ko yuktaḥ sarva bhūteṣu ko hitaḥ', 'vidvān kaḥ kaḥ samarthaḥ ca kaḥ ca eka priya darśanaḥ',
'ātmavān ko jita krodho dyutimān kaḥ anasūyakaḥ', 'kasya bibhyati devāḥ ca jāta roṣasya saṃyuge',
'etat icchāmi aham śrotum param kautūhalam hi me', 'maharṣe tvam samarthosi jñātum evam vidham naram',
'śrutvā ca etat trilokajño vālmīkeḥ nārado vacaḥ','śrūyatām iti ca āmaṃtrya prahṛṣṭo vākyam abravīt',
'bahavo durlabhāḥ ca eva ye tvayā kīrtitā guṇāḥ', 'mune vakṣṣyāmi aham buddhvā taiḥ uktaḥ śrūyatām naraḥ',
'ikṣvāku vaṃśa prabhavo rāmo nāma janaiḥ śrutaḥ', 'niyata ātmā mahāvīryo dyutimān dhṛtimān vaśī',
'buddhimān nītimān vāṅgmī śrīmān śatru nibarhaṇaḥ', 'vipulāṃso mahābāhuḥ kaṃbu grīvo mahāhanuḥ']
from Data_generator import new_list
padass=padas
from aksharamukha import transliterate
st1=[[]]
with open('output.txt', 'r', encoding='utf-8') as file:
    file_contents = [line.strip() for line in file.readlines()]
#print(file_contents)
def script():
        st2=[]
        #st2.append(transliterate.process('IAST','ITRANS',padas[pada]))
        for i in new_list:
                st2.append(transliterate.process('IAST','RomanColloquial',i))
        return st2
st=[]
with open('output2.txt', 'w', encoding='utf-8') as file:
    # Write the contents of the list to the file
    file.writelines("\n".join(st))
#st=script()
#print(st)
#print(st)
def consonants(padas):
        for i in range(len(padas)):
                #print(i)
                for j in range(len(padas[i])-1):
                        if j>=len(padas[i]):
                                        break
                        if padas[i][j]=='a' or padas[i][j]=='e' or  padas[i][j]=='i' or padas[i][j]=='o' or padas[i][j]=='u' or padas[i][j]==' ':
                                padas[i]=padas[i].replace(padas[i][j],'')
                               # print('condition met')
                               
        return padas
#consonants(padas)

#print(consonants(['I am a good boy']))
