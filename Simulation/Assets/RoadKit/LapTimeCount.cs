using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class LapTimeCount : MonoBehaviour
{
    public static int Min;
    public static int Sec;
    public static int Mils;
    public float RaceTime = 0;
    public static string TimeDisplay;

   
    public Text timeText;
    public GameObject TimeBox;
    // Update is called once per frame
    void Update()
    {
        if (Global.StartTimer == true)
        {
            RaceTime += Time.deltaTime;
            Min = Mathf.FloorToInt(RaceTime / 60);
            Sec = Mathf.FloorToInt(RaceTime % 60);
            //Mils = Mathf.FloorToInt(RaceTime % 3600);
            timeText.text = string.Format("Time {0:00}:{1:00}", Min, Sec);
        }
    }

    void OnTriggerEnter()
    {
        if (Global.StartTimer == false)
        {
            Debug.Log("Start timer");
            Global.StartTimer = true;
        }
        else
        {
            Debug.Log("End Timer");
            Global.StartTimer = false;
        }
  
    }

}
