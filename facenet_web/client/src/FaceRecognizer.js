import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiRotateCw } from 'react-icons/fi';

const API_URL = '/api/recognize/';

// 타이핑 효과
const TypingText = ({ text, speed = 10 }) => {
    const [displayedText, setDisplayedText] = useState('');
    const [index, setIndex] = useState(0);

    useEffect(() => {
        if (index < text.length) {
            const timeout = setTimeout(() => {
                setDisplayedText((prev) => prev + text.charAt(index));
                setIndex((prev) => prev + 1);
            }, speed);
            return () => clearTimeout(timeout);
        }
    }, [text, speed, index]);

    return <span>{displayedText}</span>;
};

// 분석 로딩 화면
const LoadingScreen = () => (
    <div style={loadingStyles.container}>
        <div style={loadingStyles.spinner}></div>
        <TypingText text="이미지 스캔 및 생체 정보 대조 진행 중..." speed={40} />
        <p style={loadingStyles.subtext}>서버 응답 대기 중 (STATUS 200/503)</p>
    </div>
);

// 로딩 스타일
const loadingStyles = {
    container: {
        marginTop: '30px',
        padding: '30px',
        background: '#0a0a0a',
        border: '2px solid #0077ff',
        boxShadow: '0 0 10px #0077ff, inset 0 0 10px #0077ff',
        color: '#0077ff',
        textAlign: 'center',
        fontFamily: 'monospace',
        borderRadius: '5px',
    },
    spinner: {
        border: '4px solid rgba(0, 255, 255, 0.2)',
        borderTop: '4px solid #0077ff',
        borderRadius: '50%',
        width: '30px',
        height: '30px',
        animation: 'spin 1s linear infinite',
        margin: '0 auto 15px',
    },
    subtext: {
        marginTop: '10px',
        fontSize: '0.8em',
        color: '#0077ff',
    },
};


function FaceRecognizer() {
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const KNOWN_PERSONNEL = [
        "빌 게이츠", 
        "나렌드라 모디", 
        "마윈", 
        "일론 머스크", 
        "도널드 트럼프"
    ];

    const handleImageChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setImage(file);
            setPreview(URL.createObjectURL(file));
            setResult(null); 
            setError(null);
        }
    };
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!image) {
            alert('인식할 이미지를 선택해주세요.');
            return;
        }

        setLoading(true);
        setResult(null);
        setError(null);

        const formData = new FormData();
        formData.append('image', image); 

        try {
            const response = await axios.post(API_URL, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            setResult(response.data);
        } catch (err) {
            if (err.response && err.response.status === 429) {
                const detailMessage = err.response.data.detail || "서버에서 제공한 자세한 메시지가 없습니다.";
                
                // 429 에러 발생 시 Alert 호출
                alert(`🚨 호출 횟수 제한에 걸렸습니다. 잠시 후 다시 시도해 주세요. (${detailMessage})`);
                
                setLoading(false); // 로딩 종료
                return; // Alert 후 함수 종료 (아래의 일반 에러 로직을 실행하지 않음)
            }
            // ----------------------------------------------------

            console.error('API 요청 실패:', err.response ? err.response.data : err.message);
            
            // 429가 아닌 다른 에러 (400, 500 등)를 처리하는 기존 로직
            const errorMessage = err.response && err.response.data && err.response.data.message
                ? err.response.data.message
                : '얼굴 인식 서버와의 통신에 실패했습니다.';
                
            setError(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    // 결과 표시
    const renderResult = () => {
        if (!result) return null;

        const { result: status, name, distance, threshold, raw_profile_data } = result;
        
        // 숫자형 변환
        const numericDistance = parseFloat(distance);
        const numericThreshold = parseFloat(threshold);

        // 근접 미확인 판단을 위한 완충 범위 (임계값 + 0.1)
        const CLOSE_MISS_BUFFER = 0.1;

        // 타임라인 상세 정보를 렌더링
        const renderTimeline = (timeline) => {
            if (!timeline || Object.keys(timeline).length === 0) return <li>[SYSTEM] 활동 기록 데이터 없음</li>;

            const keyMap = {
                'birth': '출생',
                'childhood': '유년기/성장배경',
                'education': '교육',
                'early_career': '초기 경력',
                'turning_point': '전환점',
                'major_achievements': '주요 업적',
                'recent': '최근 활동'
            };

            return Object.entries(timeline).map(([key, value]) => {
                const segments = value.split(/;|\n/).map(seg => seg.trim()).filter(Boolean);

                return (
                    <li key={key} style={{marginLeft: '20px', textAlign: 'left', lineHeight: '1.5'}}>
                        <div style={{display: 'flex', flexDirection: 'row', flexWrap: 'wrap'}}>
                            <strong style={{color: '#fff', flexShrink: 0, marginRight: '5px'}}>
                                {keyMap[key] ? `${keyMap[key]}:` : key.toUpperCase() + ':'}
                            </strong>
                            <div style={{flex: 1}}>
                                {segments.map((seg, idx) => (
                                    <div key={idx}>{seg}</div>
                                ))}
                            </div>
                        </div>
                    </li>
                );
            });
        };
        
        // 사건(배열) 상세 정보를 렌더링
        const renderMilestones = (milestones) => {
            if (!milestones || milestones.length === 0) return <li>[SYSTEM] 주요 사건 데이터 없음</li>;
            return milestones.map((item, index) => (
                <li key={index} style={{marginLeft: '20px', textAlign: 'left'}}>
                    <strong style={{color: '#fff'}}> {item.year}:</strong> {item.event}
                </li>
            ));
        };

        // SF 스타일 컨테이너
        const sfContainerStyle = {
            marginTop: '20px', 
            border: '2px solid #0077ff', 
            padding: '15px', 
            borderRadius: '5px', 
            background: '#0a0a0a', 
            color: '#0077ff', 
            textAlign: 'left',
            fontFamily: 'monospace',
            lineHeight: '1.4',
            boxShadow: '0 0 10px #0077ffc7',
        };
        
        // 식별 성공 [Success]
        if (status === 'success') {
            const fullProfile = raw_profile_data || {}; 
            const displayName = fullProfile.full_name_ko || name;

            return (
                <div style={sfContainerStyle}>
                    <h3 style={{color: '#00ff00', borderBottom: '1px solid #00ff00', paddingBottom: '8px', fontSize: '18px', textAlign: 'center'}}>
                        <TypingText 
                            // 식별 성공: 대상 확인 - [이름]
                            text={`분석 완료: 식별 결과 - ${displayName.toUpperCase()}`} 
                            speed={30} 
                        />
                    </h3>
                    
                    <p style={{marginTop: '10px', textAlign: 'center'}}>
                        [결과 코드: 200 OK | 매칭 거리: **{distance}** (임계값: {threshold})]
                    </p>
                    <hr style={{margin: '10px 0', borderColor: '#0077ff'}}/>

                    <h4 style={{fontSize: '20px', marginBottom: '20px', color: '#0077ff', textAlign: 'center'}}>
                        <TypingText text="프로필 데이터 추출 결과" speed={50} /><br/>
                        <TypingText text="(PROFILE DATA EXTRACTION COMPLETE)" speed={50} />
                    </h4>
                    
                    <ul style={{listStyleType: 'none', paddingLeft: '0'}}>
                        
                        {/* 기본 정보 - 이름, 출생지, 거주지 */}
                        <li style={{ color: '#fff' }}>
                            <strong style={{color: '#0077ff'}}>전체 이름 (KO/EN):</strong> {fullProfile.full_name_ko || 'N/A'} / {fullProfile.full_name_en || 'N/A'}
                        </li>
                        <li style={{marginTop: '10px', color: '#fff'}}>
                            <strong style={{color: '#0077ff'}}>국적/출생지:</strong> {fullProfile.nationality?.country || 'N/A'} ({fullProfile.nationality?.state_city || 'N/A'})
                        </li>
                        <li style={{ color: '#fff' }}>
                            <strong style={{color: '#0077ff'}}>현재 거주지:</strong> {fullProfile.residence?.country || 'N/A'} ({fullProfile.residence?.state_city || 'N/A'})
                        </li>
                        
                        {/* 소속, 직업 */}
                        <li style={{marginTop: '10px'}}>
                            <strong style={{color: '#0077ff'}}>주요 소속/직업:</strong>
                            <ul style={{listStyleType: 'square', paddingLeft: '20px', marginTop: '5px', fontSize: '0.95em', color: '#fff'}}>
                                {(fullProfile.affiliations || []).map((aff, index) => <li key={index}>{aff}</li>)}
                                {(fullProfile.affiliations || []).length === 0 && <li>[SYSTEM] 활성 소속 정보 없음</li>}
                            </ul>
                        </li>
                        
                        {/* 인물에 대한 태그 또는 키워드 */}
                        <li style={{marginTop: '10px', color: '#fff' }}>
                            <strong style={{color: '#0077ff'}}>키워드/태그:</strong> {(fullProfile.keywords || []).join(' | ') || '[SYSTEM] 키워드 정보 없음'}
                        </li>

                        {/* 타임라인 */}
                        <li style={{marginTop: '15px'}}>
                            <strong style={{color: '#0077ff'}}>타임라인 (TIMELINE LOG):</strong>
                            <ul style={{listStyleType: 'none', paddingLeft: '0', marginTop: '5px', fontSize: '15px', lineHeight: '1.4', color: '#fff'}}>
                                {renderTimeline(fullProfile.timeline)}
                            </ul>
                        </li>

                        {/* 주요 사건 */}
                        <li style={{marginTop: '15px'}}>
                            <strong style={{color: '#0077ff'}}>주요 사건 (MILESTONE EVENTS):</strong>
                            <ul style={{listStyleType: 'disc', paddingLeft: '20px', marginTop: '5px', fontSize: '0.9em', color: '#fff'}}>
                                {renderMilestones(fullProfile.milestones)}
                            </ul>
                        </li>
                    </ul>

                </div>
            );

        // 미식별 [Unknown]
        } else if (status === 'unknown') {
            let detailedMessage = "";
            let warningLevel = "";
            
            // 미식별 상태일 때, 거리가 임계값 + Buffer 이내인 경우
            if (numericDistance <= numericThreshold + CLOSE_MISS_BUFFER) {
                warningLevel = "근접 미확인 (CLOSE UNKNOWN)";
                detailedMessage = `매칭 거리 (${numericDistance.toFixed(4)})가 임계값 (${numericThreshold})에 **매우 근접**하나 초과하였습니다. 추가 분석으로 식별 가능성이 존재합니다.`;
            } 
            // 미식별 상태일 때, 거리가 임계값+Buffer를 초과한 경우
            else {
                warningLevel = "원거리 미확인 (FAR UNKNOWN)";
                detailedMessage = `매칭 거리 (${numericDistance.toFixed(4)})가 임계값 (${numericThreshold})보다 **크게 초과**하여 불일치합니다.`;
            }


            return (
                <div style={{...sfContainerStyle, borderColor: '#ffcc00', boxShadow: '0 0 10px rgba(255, 204, 0, 0.7)', color: '#ffcc00'}}>
                    <h3>{warningLevel}</h3>
                    <p>
                        {detailedMessage}
                    </p>
                    <p style={{marginTop: '10px'}}>
                    [SYSTEM] 접근 거부됨. 해당 인물 데이터가 부족합니다.
                    </p>
                </div>
            );

        // 3. 얼굴 미감지 (Not Found)
        // ------------------------------------------------
        } else if (status === 'not_found') {
            return (
                <div style={{...sfContainerStyle, borderColor: '#ff0000', boxShadow: '0 0 10px rgba(255, 0, 0, 0.7)', color: '#ff0000'}}>
                    <h3>얼굴 감지 실패 (FACE DETECTION FAILURE)</h3>
                    <p>이미지 프레임 내에서 유효한 인물 데이터가 감지되지 않았습니다.</p>
                </div>
            );
        // 4. 서버 초기화/내부 오류 (Error)
        } else if (status === 'error') {
            return (
                <div style={{...sfContainerStyle, borderColor: '#ff0000', boxShadow: '0 0 10px rgba(255, 0, 0, 0.7)', color: '#ff0000'}}>
                    <h3>시스템 오류 [CODE 500/503]</h3>
                    <p>치명적인 백엔드 장애가 발생했습니다. 로그를 확인하십시오.</p>
                    <p>메시지: {result.message}</p>
                </div>
            );
        }
    };
    
    // [최종 렌더링]
    return (
        <div style={{ 
            maxWidth: '870px', 
            margin: '50px auto', 
            padding: '20px', 
            border: '2px solid #333', 
            borderRadius: '10px',
            background: '#222',
            color: '#fff',
            fontFamily: 'Arial, sans-serif'
        }}>
            <h1 style={{color: '#0077ff', borderBottom: '1px solid #0077ff', paddingBottom: '10px'}}>
                BIOMETRIC IDENTIFICATION SYSTEM V1.0
            </h1>
            <p style={{color: '#ccc', marginBottom: '20px', textAlign: 'center'}}>
                <TypingText 
                    text={`현재 학습된 인물 : [${KNOWN_PERSONNEL.join(', ')}]`} 
                    speed={20} 
                />
            </p>
            
            {/* 파일 선택 및 버튼 정렬 - Flexbox를 사용하여 양 끝에 배치 */}
            <form onSubmit={handleSubmit} style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                marginBottom: '20px' 
            }}>
                <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleImageChange} 
                    disabled={loading}
                    style={{ padding: '10px', border: '1px solid #0077ff', background: '#0a0a0a', color: '#0077ff' }}
                />

                <div style={{ display: 'flex', gap: '10px' }}>
                    {/* 리셋 버튼 */}
                    <button
                        type="button"
                        onClick={() => {
                            setImage(null);
                            setPreview(null);
                            setResult(null);
                            setError(null);
                            setLoading(false);
                        }}
                        style={{
                            padding: '10px 20px',
                            background: '#0077ff',
                            color: '#fff',
                            border: 'none',
                            borderRadius: '5px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '5px',
                            fontWeight: 'bold',
                        }}
                    >
                        {/* 반시계 90도 회전 */}
                        <FiRotateCw style={{ transform: 'rotate(-90deg)' }} /> 
                        RESET
                    </button>

                    {/* 기존 스캔 버튼 */}
                    <button 
                        type="submit" 
                        disabled={!image || loading}
                        style={{ 
                            padding: '10px 20px', 
                            background: loading ? '#333' : '#0077ff', 
                            color: loading ? '#999' : '#fff', 
                            border: 'none', 
                            borderRadius: '5px', 
                            cursor: 'pointer', 
                            transition: 'background 0.3s' 
                        }}
                    >
                        {loading ? 'PROCESSING...' : 'INITIATE SCAN'}
                    </button>
                </div>
            </form>

            {error && (
                <div style={{ color: 'darkred', background: '#ffe0e0', padding: '10px', marginTop: '20px', borderRadius: '4px' }}>
                    **[시스템 경고]** {error}
                </div>
            )}

            {/* 이미지 미리보기 중앙 정렬 */}
            {preview && (
                <div style={{ 
                    marginTop: '30px', 
                    border: '1px solid #0077ff', 
                    padding: '10px', 
                    background: '#0a0a0a',
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    textAlign: 'center', 
                }}>
                    <h3 style={{color: '#0077ff'}}>UPLOADED IMAGE PREVIEW</h3>
                    <img 
                        src={preview} 
                        alt="Target Image" 
                        style={{ 
                            maxWidth: '100%', 
                            maxHeight: '300px', 
                            border: '1px solid #0077ff',
                            display: 'block', 
                            margin: '10px 0' 
                        }}
                    />
                </div>
            )}

            {/* 로딩 화면 렌더링 */}
            {loading && <LoadingScreen />}

            {/* 결과 화면 렌더링 */}
            {!loading && renderResult()}
            
        </div>
    );
}

export default FaceRecognizer;