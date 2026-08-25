(()=>{
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const finePointer=window.matchMedia('(pointer: fine)').matches;
  document.body.classList.add('motion-enabled');

  const progress=document.querySelector('#pageScrollProgress i');
  let scrollFrame=0;
  const paintScroll=()=>{
    scrollFrame=0;
    if(!progress)return;
    const range=document.documentElement.scrollHeight-window.innerHeight;
    const ratio=range>0?Math.min(1,Math.max(0,window.scrollY/range)):0;
    progress.style.transform=`scaleX(${ratio})`;
  };
  const scheduleScroll=()=>{if(!scrollFrame)scrollFrame=requestAnimationFrame(paintScroll)};
  window.addEventListener('scroll',scheduleScroll,{passive:true});
  window.addEventListener('resize',scheduleScroll,{passive:true});
  paintScroll();

  const revealTargets=[...document.querySelectorAll('[data-reveal],.trust-panel,.analysis-quality,.performance-card,#report-scorecard,#report-coaching>*,#report-training>*,.replay-callout')];
  document.querySelectorAll('[data-motion-sequence]').forEach(sequence=>{
    [...sequence.children].forEach((element,index)=>{
      element.dataset.motionSequenceItem='true';
      element.style.setProperty('--wiq-reveal-order',String(Math.min(index,4)));
      revealTargets.push(element);
    });
  });
  revealTargets.forEach((element,index)=>{
    element.classList.add('wiq-reveal');
    if(!element.style.getPropertyValue('--wiq-reveal-order'))element.style.setProperty('--wiq-reveal-order',String(index%4));
  });
  if(reduced||!('IntersectionObserver'in window)){
    revealTargets.forEach(element=>element.classList.add('is-visible'));
  }else{
    const revealObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(!entry.isIntersecting)return;
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }),{rootMargin:'0px 0px -6% 0px',threshold:.06});
    revealTargets.forEach(element=>revealObserver.observe(element));
  }

  const parseStat=element=>{
    if(element.children.length)return null;
    const text=element.textContent.trim();
    const match=text.match(/^(-?\d+(?:\.\d+)?)(%|s)?$/);
    if(!match)return null;
    return{value:Number(match[1]),suffix:match[2]||'',decimals:(match[1].split('.')[1]||'').length};
  };
  const countTo=(element,nextValue,options={})=>{
    if(!element)return;
    const target=Number(nextValue);
    if(!Number.isFinite(target))return;
    const suffix=options.suffix??'';
    const decimals=Number.isInteger(options.decimals)?options.decimals:0;
    const currentValue=Number(element.dataset.wiqValue);
    const parsedCurrent=parseStat(element)?.value;
    const from=Number.isFinite(options.from)?Number(options.from):(Number.isFinite(currentValue)?currentValue:(Number.isFinite(parsedCurrent)?parsedCurrent:target));
    const changed=from!==target;
    element.dataset.wiqValue=String(target);
    element.dataset.wiqAnimation=String(Number(element.dataset.wiqAnimation||0)+1);
    const animation=element.dataset.wiqAnimation;
    const finish=()=>{element.textContent=target.toFixed(decimals)+suffix;if(options.impact&&changed&&!reduced){const surface=element.closest('.metric,.performance-stat,.live-stat-grid>div');if(surface){surface.classList.remove('is-stat-impact');requestAnimationFrame(()=>surface.classList.add('is-stat-impact'))}}};
    if(reduced||!changed){finish();return;}
    const start=performance.now(),duration=Math.max(160,Math.min(650,Number(options.duration)||420));
    const tick=now=>{
      if(element.dataset.wiqAnimation!==animation)return;
      const t=Math.min(1,(now-start)/duration),eased=1-Math.pow(1-t,3);
      element.textContent=(from+(target-from)*eased).toFixed(decimals)+suffix;
      if(t<1)requestAnimationFrame(tick);else finish();
    };
    requestAnimationFrame(tick);
  };
  const animateStat=element=>{
    const stat=parseStat(element);
    if(!stat||element.dataset.wiqCounted)return;
    element.dataset.wiqCounted='true';
    countTo(element,stat.value,{from:0,suffix:stat.suffix,decimals:stat.decimals,duration:520,impact:true});
  };
  const statTargets=[...document.querySelectorAll('body[data-page=result] .analysis-quality-grid strong,body[data-page=result] .performance-stat>strong,body[data-page=result] .metric .value,body[data-page=result] .score,body[data-page=result] #report-scorecard tbody td:nth-child(2),body[data-page=result] #report-scorecard tbody td:nth-child(3),body[data-page=dashboard] .metric .value')];
  if('IntersectionObserver'in window&&!reduced){
    const statObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(!entry.isIntersecting)return;
      animateStat(entry.target);statObserver.unobserve(entry.target);
    }),{threshold:.35});
    statTargets.forEach(element=>statObserver.observe(element));
  }else{
    statTargets.forEach(animateStat);
  }

  const primaryActions=[...document.querySelectorAll('[data-motion-primary]')];
  primaryActions.forEach(element=>element.classList.add('wiq-primary-motion'));
  if(!reduced&&finePointer){
    primaryActions.forEach(element=>{
      element.addEventListener('pointermove',event=>{
        const box=element.getBoundingClientRect();
        const x=Math.max(-4,Math.min(4,(event.clientX-(box.left+box.width/2))*.075));
        const y=Math.max(-3,Math.min(3,(event.clientY-(box.top+box.height/2))*.075));
        element.style.setProperty('--wiq-magnet-x',`${x.toFixed(2)}px`);
        element.style.setProperty('--wiq-magnet-y',`${y.toFixed(2)}px`);
        element.style.setProperty('--wiq-glare-x',`${Math.max(0,Math.min(100,(event.clientX-box.left)/box.width*100)).toFixed(1)}%`);
      });
      element.addEventListener('pointerleave',()=>{
        element.style.setProperty('--wiq-magnet-x','0px');
        element.style.setProperty('--wiq-magnet-y','0px');
      });
    });
  }

  const reportLinks=[...document.querySelectorAll('.report-path a[href^="#"]')];
  if(reportLinks.length&&'IntersectionObserver'in window){
    const linkByTarget=new Map(reportLinks.map(link=>[link.getAttribute('href').slice(1),link]));
    const sectionObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(!entry.isIntersecting)return;
      reportLinks.forEach(link=>link.removeAttribute('aria-current'));
      linkByTarget.get(entry.target.id)?.setAttribute('aria-current','true');
    }),{rootMargin:'-24% 0px -62% 0px',threshold:0});
    linkByTarget.forEach((link,id)=>{const section=document.getElementById(id);if(section)sectionObserver.observe(section)});
  }

  const impact=element=>{
    if(!element||reduced)return;
    element.classList.remove('selection-impact');
    requestAnimationFrame(()=>element.classList.add('selection-impact'));
  };
  window.WarriorIQMotion={impact,reduced,countTo};
})();
